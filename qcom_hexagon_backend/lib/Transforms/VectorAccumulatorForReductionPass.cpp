//===- VectorAccumulatorForReductionPass.cpp ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass is very specific to improving a pattern of IR found in block
// softmax. Its not super generalized and that is left as future work.
//
// Replaces per-tile scalar accumulator in tiled reduction SCF loops with a
// vector accumulator, deferring the single horizontal reduction to after the
// inner loop.  This eliminates N-1 serialising horizontal reductions per row
// (where N = number of tiles).
//
// Pattern matched (per row):
//
//   scf.for %row = 0 to Rows step 1                        <- outer, step == 1
//   iter_args(%acc = %init) -> tensor<...xT> {
//
//     scf.for %col = 0 to Cols step S                      <- inner, step S
//     iter_args(%v = %acc) -> tensor<...xT> {
//       %tile = tensor.extract_slice %input[...][...,S][..]  <- 1-D, width S
//       %prev = tensor.extract_slice %v[...][..1..][..]      <- rank-0 scalar
//       %res  = linalg.generic {reduction} ins(%tile) outs(%prev) -> tensor<T>
//       %nv   = tensor.insert_slice %res into %v[...][..1..][..]
//       scf.yield %nv
//     }
//     scf.yield ...
//   }
//
// Constraints for the pattern to fire:
//   - Both inner and outer loops have exactly one iter-arg.
//   - The inner loop's init arg equals the outer loop's region iter-arg.
//   - The outer loop's yield directly returns the inner loop's result.
//   - The inner loop step S is a statically known positive constant.
//   - The reduction linalg.generic has exactly one 1-D input of width S and
//     one rank-0 (scalar) output.
//   - The binary op is addf, maxnumf, maximumf, minnumf, or minimumf.
//   - Single-use invariants on intermediate values (guard rails, below).
//
// Guard rails (all checked in matchReductionLoop):
//   [R1] innerFor.getResult(0) has exactly one use (the outer yield).
//   [R2] accSliceOp result has exactly one use (the reductionOp).
//   [R3] reductionOp result has exactly one use (the insertSliceOp).
//   [R4] insertSliceOp result has exactly one use (the scf.yield).
//   [R5] reductionOp ins/outs are the matched data tile and accumulator slice.
//   [R6] insertSliceOp writes the reductionOp result back.
//   [R7] inner-loop yield directly forwards insertSliceOp.
//   Each of inputSliceOp / accSliceOp / reductionOp / insertSliceOp must be
//   the unique candidate in the inner body; a duplicate rejects the match.
//
// Generalisation:
//   - The accumulator tensor may have any rank (not just 1-D).  The
//     accSliceOp must still extract a rank-0 (scalar) slice.
//   - Any float element type (f16, bf16, f32, f64, …) is supported.
//   - The tile width S (inner step) may be any constant > 1.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_VECTORACCUMULATORFORREDUCTION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static std::optional<int64_t> getIntConst(Value v) {
  if (auto op = v.getDefiningOp<arith::ConstantIndexOp>())
    return op.value();
  if (auto op = v.getDefiningOp<arith::ConstantIntOp>())
    return op.value();
  return std::nullopt;
}

// Returns the floating-point identity element for the binary op, or nullopt
// if the op is not a supported reduction kind.
static std::optional<FloatAttr> getReductionIdentity(Operation *binOp,
                                                     FloatType fType) {
  const auto &sem = fType.getFloatSemantics();
  if (isa<arith::MaxNumFOp, arith::MaximumFOp>(binOp))
    return FloatAttr::get(fType, APFloat::getInf(sem, /*Negative=*/true));

  if (isa<arith::MinNumFOp, arith::MinimumFOp>(binOp))
    return FloatAttr::get(fType, APFloat::getInf(sem, /*Negative=*/false));

  if (isa<arith::AddFOp>(binOp))
    return FloatAttr::get(fType, APFloat::getZero(sem));
  return std::nullopt;
}

struct ReductionLoopInfo {
  scf::ForOp outerFor;
  scf::ForOp innerFor;
  tensor::ExtractSliceOp inputSliceOp; // 2-D…→1-D data tile
  tensor::ExtractSliceOp accSliceOp;   // accumulator tensor → rank-0 scalar
  linalg::GenericOp reductionOp;       // the per-tile reduction generic
  tensor::InsertSliceOp insertSliceOp; // scalar written back into accumulator
  Operation *binOp = nullptr;          // binary op inside reduction body
  FloatAttr identityAttr;              // identity element for the reduction
};

// Find suitable candidates.
static std::optional<ReductionLoopInfo>
matchReductionLoop(scf::ForOp innerFor) {

  // ── Basic structure of the inner loop ────────────────────────────────────
  // Single iter arg that is a ranked tensor of float element type, and
  // [R1] inner result must have exactly one use (the outer yield).
  // Inner step must be a positive constant (= tile / vector width S).
  if (innerFor.getInitArgs().size() != 1)
    return std::nullopt;

  auto iterArgType =
      dyn_cast<RankedTensorType>(innerFor.getInitArgs()[0].getType());
  if (!iterArgType)
    return std::nullopt;

  auto fType = dyn_cast<FloatType>(iterArgType.getElementType());
  if (!fType)
    return std::nullopt;

  if (!innerFor.getResult(0).hasOneUse())
    return std::nullopt;

  auto innerStep = getIntConst(innerFor.getStep());
  if (!innerStep || *innerStep <= 1)
    return std::nullopt;
  int64_t vecSize = *innerStep;

  // ── Outer loop constraints ────────────────────────────────────────────────
  // Outer step must be 1 (iterates one row at a time).
  // Exactly one iter-arg on the outer loop.
  // The inner loop is seeded from the outer loop's live iter-arg.
  // The outer yield directly forwards the inner loop's result.

  auto outerFor = dyn_cast<scf::ForOp>(innerFor->getParentOp());
  if (!outerFor)
    return std::nullopt;

  if (!getIntConst(outerFor.getStep()) || *getIntConst(outerFor.getStep()) != 1)
    return std::nullopt;

  if (outerFor.getInitArgs().size() != 1)
    return std::nullopt;

  if (innerFor.getInitArgs()[0] != outerFor.getRegionIterArgs()[0])
    return std::nullopt;

  auto outerYield =
      cast<scf::YieldOp>(outerFor.getRegion().front().getTerminator());
  if (outerYield.getNumOperands() != 1 ||
      outerYield.getOperands()[0] != innerFor.getResult(0))
    return std::nullopt;

  // ── Match ops inside the inner loop body ─────────────────────────────────
  // Each role is picked at most once; a second matching op rejects the
  // candidate.  accSliceOp must read from the inner iter-arg and
  // insertSliceOp must write back into it, so the rewrite (which uses
  // accSliceOp's offsets/sizes/strides on the outer iter-arg) is sound.

  ReductionLoopInfo info;
  info.outerFor = outerFor;
  info.innerFor = innerFor;
  Value innerIterArg = innerFor.getRegionIterArgs()[0];

  for (Operation &op : innerFor.getRegion().front()) {
    if (auto eslOp = dyn_cast<tensor::ExtractSliceOp>(&op)) {
      auto rt = dyn_cast<RankedTensorType>(eslOp.getResult().getType());
      if (!rt)
        continue;
      // 1-D tile of exactly vecSize elements → data slice.
      if (rt.getRank() == 1 && rt.getShape()[0] == vecSize) {
        if (info.inputSliceOp)
          return std::nullopt;
        info.inputSliceOp = eslOp;
      }
      // Rank-0 scalar from the inner iter-arg → accumulator slice.
      else if (rt.getRank() == 0 && eslOp.getSource() == innerIterArg) {
        if (info.accSliceOp)
          return std::nullopt;
        info.accSliceOp = eslOp;
      }
    } else if (auto genOp = dyn_cast<linalg::GenericOp>(&op)) {
      auto iters = genOp.getIteratorTypesArray();
      if (iters.size() == 1 && iters[0] == utils::IteratorType::reduction &&
          genOp.getInputs().size() == 1 && genOp.getOutputs().size() == 1) {
        auto inTy = dyn_cast<RankedTensorType>(genOp.getInputs()[0].getType());
        auto outTy =
            dyn_cast<RankedTensorType>(genOp.getOutputs()[0].getType());
        if (inTy && inTy.getRank() == 1 && inTy.getShape()[0] == vecSize &&
            outTy && outTy.getRank() == 0) {
          if (info.reductionOp)
            return std::nullopt;
          info.reductionOp = genOp;
        }
      }
    } else if (auto insOp = dyn_cast<tensor::InsertSliceOp>(&op)) {
      // Must write back into the inner iter-arg (i.e., be on the carry chain).
      if (insOp.getDest() != innerIterArg)
        continue;
      if (info.insertSliceOp)
        return std::nullopt;
      info.insertSliceOp = insOp;
    }
  }

  if (!info.inputSliceOp || !info.accSliceOp || !info.reductionOp ||
      !info.insertSliceOp)
    return std::nullopt;

  // ── Guard rails on intermediate uses [R2–R4] ─────────────────────────────
  // [R2] accSliceOp feeds only the reductionOp.
  // [R3] reductionOp result feeds only the insertSliceOp.
  // [R4] insertSliceOp result feeds only the scf.yield.
  //
  if (!info.accSliceOp.getResult().hasOneUse())
    return std::nullopt;
  if (!info.reductionOp.getResult(0).hasOneUse())
    return std::nullopt;
  if (!info.insertSliceOp.getResult().hasOneUse())
    return std::nullopt;

  // ── Cross-links between the four roles [R5–R7] ───────────────────────────
  // [R5] reductionOp ins/outs are the matched data tile and accumulator slice.
  // [R6] insertSliceOp writes the reductionOp result back.
  // [R7] inner-loop yield forwards insertSliceOp.
  //
  if (info.reductionOp.getInputs()[0] != info.inputSliceOp.getResult() ||
      info.reductionOp.getOutputs()[0] != info.accSliceOp.getResult())
    return std::nullopt;
  if (info.insertSliceOp.getSource() != info.reductionOp.getResult(0))
    return std::nullopt;
  auto innerYield =
      cast<scf::YieldOp>(innerFor.getRegion().front().getTerminator());
  if (innerYield.getOperand(0) != info.insertSliceOp.getResult())
    return std::nullopt;

  // ── Reduction body: identify binary op and derive identity ───────────────
  for (Operation &op : info.reductionOp.getRegion().front()) {
    if (!isa<linalg::YieldOp>(&op)) {
      info.binOp = &op;
      break;
    }
  }
  if (!info.binOp)
    return std::nullopt;

  auto identity = getReductionIdentity(info.binOp, fType);
  if (!identity)
    return std::nullopt;
  info.identityAttr = *identity;

  return info;
}

// Modify the scf.for based on reduction loop info.
static void transformReductionLoop(IRRewriter &rewriter,
                                   ReductionLoopInfo info) {
  Location loc = info.innerFor.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  auto vecTy =
      cast<RankedTensorType>(info.reductionOp.getInputs()[0].getType());
  Type elemTy = vecTy.getElementType();
  int64_t vecSize = vecTy.getShape()[0];
  auto scalarTy = RankedTensorType::get({}, elemTy);

  // ── Before the inner loop: create the vector accumulator ─────────────────
  rewriter.setInsertionPoint(info.innerFor);

  Value identConst =
      arith::ConstantOp::create(rewriter, loc, info.identityAttr);

  Value vecEmpty = tensor::EmptyOp::create(rewriter, loc,
                                           ArrayRef<int64_t>{vecSize}, elemTy);
  Value vecInit = linalg::FillOp::create(rewriter, loc, ValueRange{identConst},
                                         ValueRange{vecEmpty})
                      .getResult(0);

  // ── New inner loop: elementwise (parallel) accumulation ──────────────────
  Value oldInnerIV = info.innerFor.getInductionVar();
  auto newInnerFor = scf::ForOp::create(
      rewriter, loc, info.innerFor.getLowerBound(),
      info.innerFor.getUpperBound(), info.innerFor.getStep(),
      ValueRange{vecInit},
      [&](OpBuilder &b, Location loc2, Value newIV, ValueRange args) {
        Value vecAcc = args[0];

        // Clone the data-tile extract_slice, remapping the inner IV.
        IRMapping ivMap;
        ivMap.map(oldInnerIV, newIV);
        Value inputSlice =
            b.clone(*info.inputSliceOp.getOperation(), ivMap)->getResult(0);

        // Parallel elementwise generic: same binary op body, no reduction.
        auto identMap = AffineMap::getMultiDimIdentityMap(1, ctx);
        SmallVector<AffineMap> maps = {identMap, identMap};
        SmallVector<utils::IteratorType> iters = {
            utils::IteratorType::parallel};

        Block *origBody = info.reductionOp.getBody();
        auto elemOp = linalg::GenericOp::create(
            b, loc2, TypeRange{vecTy}, ValueRange{inputSlice},
            ValueRange{vecAcc}, maps, iters,
            [&](OpBuilder &bb, Location, ValueRange bArgs) {
              IRMapping bMap;
              bMap.map(origBody->getArgument(0), bArgs[0]);
              bMap.map(origBody->getArgument(1), bArgs[1]);
              for (Operation &bodyOp : *origBody)
                bb.clone(bodyOp, bMap);
            });

        scf::YieldOp::create(b, loc2, elemOp.getResult(0));
      });

  // ── After the inner loop: single final horizontal reduction ──────────────
  rewriter.setInsertionPointAfter(newInnerFor);
  Value vecResult = newInnerFor.getResult(0);

  Value scalarEmpty =
      tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{}, elemTy);
  Value scalarInit =
      linalg::FillOp::create(rewriter, loc, ValueRange{identConst},
                             ValueRange{scalarEmpty})
          .getResult(0);

  auto emptyMap = AffineMap::get(1, 0, {}, ctx);
  auto identMap1D = AffineMap::getMultiDimIdentityMap(1, ctx);
  SmallVector<AffineMap> redMaps = {identMap1D, emptyMap};
  SmallVector<utils::IteratorType> redIters = {utils::IteratorType::reduction};

  Block *origBody = info.reductionOp.getBody();
  auto finalRed = linalg::GenericOp::create(
      rewriter, loc, TypeRange{scalarTy}, ValueRange{vecResult},
      ValueRange{scalarInit}, redMaps, redIters,
      [&](OpBuilder &bb, Location, ValueRange bArgs) {
        IRMapping bMap;
        bMap.map(origBody->getArgument(0), bArgs[0]);
        bMap.map(origBody->getArgument(1), bArgs[1]);
        for (Operation &bodyOp : *origBody)
          bb.clone(bodyOp, bMap);
      });

  // Insert the scalar result back into the outer accumulator at [row].
  Value scalarResult = finalRed.getResult(0);
  Value outerIterArg = info.outerFor.getRegionIterArgs()[0];

  Value newAcc =
      tensor::InsertSliceOp::create(rewriter, loc, scalarResult, outerIterArg,
                                    info.accSliceOp.getMixedOffsets(),
                                    info.accSliceOp.getMixedSizes(),
                                    info.accSliceOp.getMixedStrides())
          .getResult();

  // Replace all uses of the old inner loop's result and erase it.
  rewriter.replaceAllUsesWith(info.innerFor.getResult(0), newAcc);
  rewriter.eraseOp(info.innerFor);
}

/// The vector-accumulator-for-reduction pass.
struct VectorAccumulatorForReductionPass
    : public ::impl::VectorAccumulatorForReductionBase<
          VectorAccumulatorForReductionPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    // walk the func to discover scf::For candidates.
    SmallVector<ReductionLoopInfo> toTransform;
    func.walk([&](scf::ForOp innerFor) {
      if (auto info = matchReductionLoop(innerFor))
        toTransform.push_back(*info);
    });

    // rewrite to vector accumulate reduction.
    for (auto &info : toTransform)
      transformReductionLoop(rewriter, info);
  }
};

} // namespace

std::unique_ptr<Pass> hexagon::createVectorAccumulatorForReductionPass() {
  return std::make_unique<VectorAccumulatorForReductionPass>();
}
