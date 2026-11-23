//===- PromoteTiledSCFLoopToVTCMPass.cpp - Promote tiled SCF to VTCM ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This pass rewrites scf.for that has the form of a tiled loop operating on
// DDR (e.g. tiled at triton level) into a scf.for that has the form of
// extract_slice insert_slice operating over DDR and VTCM. The bufferizer can
// take the new form of scf.for and emit vtcm subview → alloc → copy triplets.
// This then can be optionally double-buffered.
//
// The original tiled.scf.for is verified not to have memory aliasing
// or inter-iteration dependencies. The checks are conservative.
// Also, the original tiled scf.for may come in different forms and some
// of those IR quirks are taken care of below.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Utils.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <functional>

#define DEBUG_TYPE "promote-tiled-scf-loop-to-vtcm"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

#define GEN_PASS_DEF_PROMOTETILEDSCFLOOPTOVTCM
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

using namespace mlir;
using namespace hexagon;

namespace {

// DDR-side input: tiled `reinterpret_cast` consumed by `to_tensor`.
struct InputBinding {
  memref::ReinterpretCastOp rc;
  bufferization::ToTensorOp toTensor;
  Value flatOffset;
};

// DDR-side output: tiled `reinterpret_cast` consumed by `materialize`.
struct OutputBinding {
  memref::ReinterpretCastOp rc;
  bufferization::MaterializeInDestinationOp materialize;
  Value flatOffset;
};

struct RegionInfo {
  SmallVector<InputBinding> inputs;
  SmallVector<OutputBinding> outputs;
};

struct BindingShape {
  RankedTensorType tileTensorType; // rank-1 or rank-2 (matches original tile)
  RankedTensorType fullTensorType; // always rank-2: tensor<UB x C x f32>
  MemRefType fullMemRefType;       // matching memref type for the rc
  int64_t tileRows;                // staticSizes[0] for rank-2, 1 for rank-1
  int64_t tileCols;
  Type elementType;
};

// Affine offset validation.
//
// To ensure soundness of the rewrite, every per-iteration rc's dynamic
// flat offset must be of the form `K * iv + base` where `base` is loop-
// invariant and `K` matches the row stride implied by the rc's tile shape.
// We compute `f(iv) - f(0)` symbolically: loop-invariant terms cancel in
// the difference, leaving the iv coefficient regardless of how `base` was
// constructed. Unknown SSA values become opaque AffineSymbolExprs and
// cancel as well — but only soundly when the opaque value is itself
// loop-invariant. If the opaque value transitively depends on `iv`, the
// symbol cancellation hides real iv dependence and the validator would
// wrongly accept a non-affine offset. `dependsOnIV` flags any such opaque
// term as iv-tainted; `buildAffineExpr` propagates that flag and
// `validateRcOffset` rejects the binding.
static bool dependsOnIV(Value v, Value iv) {
  llvm::SmallPtrSet<Value, 8> visited;
  llvm::SmallVector<Value, 8> work;
  work.push_back(v);
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (cur == iv)
      return true;
    if (!visited.insert(cur).second)
      continue;
    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      continue; // block argument other than iv: stop walk
    for (Value operand : defOp->getOperands())
      work.push_back(operand);
  }
  return false;
}

static AffineExpr buildAffineExpr(Value val, Value iv,
                                  DenseMap<Value, unsigned> &symbolTable,
                                  bool &ivTainted, MLIRContext *ctx) {
  auto asSymbol = [&](Value v) -> AffineExpr {
    if (dependsOnIV(v, iv))
      ivTainted = true;
    auto [it, inserted] = symbolTable.try_emplace(v, symbolTable.size());
    return getAffineSymbolExpr(it->second, ctx);
  };

  if (val == iv)
    return getAffineDimExpr(0, ctx);

  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return asSymbol(val);

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    if (castOp.getIn() == iv)
      return getAffineDimExpr(0, ctx);
    return buildAffineExpr(castOp.getIn(), iv, symbolTable, ivTainted, ctx);
  }

  APInt cstVal;
  if (matchPattern(val, m_ConstantInt(&cstVal)))
    return getAffineConstantExpr(cstVal.getSExtValue(), ctx);

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp))
    return buildAffineExpr(addOp.getLhs(), iv, symbolTable, ivTainted, ctx) +
           buildAffineExpr(addOp.getRhs(), iv, symbolTable, ivTainted, ctx);

  if (auto subOp = dyn_cast<arith::SubIOp>(defOp))
    return buildAffineExpr(subOp.getLhs(), iv, symbolTable, ivTainted, ctx) -
           buildAffineExpr(subOp.getRhs(), iv, symbolTable, ivTainted, ctx);

  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    APInt lc, rc;
    bool lhsConst = matchPattern(mulOp.getLhs(), m_ConstantInt(&lc));
    bool rhsConst = matchPattern(mulOp.getRhs(), m_ConstantInt(&rc));
    if (lhsConst) {
      AffineExpr rhs =
          buildAffineExpr(mulOp.getRhs(), iv, symbolTable, ivTainted, ctx);
      return getAffineConstantExpr(lc.getSExtValue(), ctx) * rhs;
    }
    if (rhsConst) {
      AffineExpr lhs =
          buildAffineExpr(mulOp.getLhs(), iv, symbolTable, ivTainted, ctx);
      return lhs * getAffineConstantExpr(rc.getSExtValue(), ctx);
    }
  }

  return asSymbol(val);
}

static std::optional<int64_t> extractDimCoeff(AffineExpr expr,
                                              unsigned dimPos) {
  if (isa<AffineConstantExpr>(expr) || isa<AffineSymbolExpr>(expr))
    return 0;
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return dim.getPosition() == dimPos ? 1 : 0;

  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;

  if (bin.getKind() == AffineExprKind::Add) {
    auto lc = extractDimCoeff(bin.getLHS(), dimPos);
    auto rc = extractDimCoeff(bin.getRHS(), dimPos);
    if (!lc || !rc)
      return std::nullopt;
    return *lc + *rc;
  }

  if (bin.getKind() == AffineExprKind::Mul) {
    auto lc = extractDimCoeff(bin.getLHS(), dimPos);
    auto rc = extractDimCoeff(bin.getRHS(), dimPos);
    if (lc && *lc == 0) {
      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        if (rc)
          return cst.getValue() * (*rc);
      if (rc && *rc != 0)
        return std::nullopt;
      return 0;
    }
    if (rc && *rc == 0) {
      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        if (lc)
          return (*lc) * cst.getValue();
      if (lc && *lc != 0)
        return std::nullopt;
      return 0;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<Value> validateRcOffset(memref::ReinterpretCastOp rc,
                                             Value iv,
                                             int64_t expectedRowStride) {
  ArrayRef<int64_t> staticOffsets = rc.getStaticOffsets();
  if (staticOffsets.empty() || staticOffsets[0] != ShapedType::kDynamic) {
    LLVM_DEBUG(DBGS() << "  rc offset is loop-invariant — rejecting\n");
    return std::nullopt;
  }
  OperandRange dyn = rc.getOffsets();
  if (dyn.empty()) {
    LLVM_DEBUG(DBGS() << "  rc offset slot is kDynamic but no operand — "
                         "rejecting\n");
    return std::nullopt;
  }
  Value flatOffset = dyn[0];

  MLIRContext *ctx = rc.getContext();
  DenseMap<Value, unsigned> symbolTable;
  bool ivTainted = false;
  AffineExpr fExpr =
      buildAffineExpr(flatOffset, iv, symbolTable, ivTainted, ctx);
  if (ivTainted) {
    LLVM_DEBUG(DBGS() << "  rc offset has iv-tainted opaque term — "
                         "rejecting\n");
    return std::nullopt;
  }

  unsigned numSymbols = symbolTable.size();
  SmallVector<AffineExpr, 4> dimReplZero = {getAffineConstantExpr(0, ctx)};
  SmallVector<AffineExpr, 4> symIdent;
  for (unsigned i = 0; i < numSymbols; ++i)
    symIdent.push_back(getAffineSymbolExpr(i, ctx));
  AffineExpr advance =
      fExpr - fExpr.replaceDimsAndSymbols(dimReplZero, symIdent);

  std::optional<int64_t> ivCoeff = extractDimCoeff(advance, /*dimPos=*/0);
  if (!ivCoeff) {
    LLVM_DEBUG(DBGS() << "  rc offset is non-affine in iv — rejecting\n");
    return std::nullopt;
  }
  if (*ivCoeff != expectedRowStride) {
    LLVM_DEBUG(DBGS() << "  rc ivCoeff=" << *ivCoeff
                      << " != expected row stride=" << expectedRowStride
                      << " — rejecting\n");
    return std::nullopt;
  }
  return flatOffset;
}

static std::optional<int64_t> getConstantValue(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  return std::nullopt;
}

// Recursively check whether `v`'s SSA-defining chain syntactically guarantees
// that `v` is divisible by `k` at runtime.  Recognises:
//   - constants whose value is divisible by k,
//   - arith.muli with at least one operand divisible by k,
//   - arith.addi with both operands divisible by k,
//   - arith.index_cast (pass-through, value preserved).
// Used to weaken the loop lb/step soundness check from `== tileRows` to
// `is a multiple of tileRows`, which still guarantees no inter-iteration
// tile overlap while accepting the dynamically-strided softmax pattern
// (e.g. step = `arith.muli %x, %c32_i32` has 32 as a factor).
static bool isStaticallyMultipleOf(Value v, int64_t k) {
  if (k == 0)
    return false;
  if (auto cst = getConstantValue(v))
    return *cst % k == 0;
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;
  if (auto mul = dyn_cast<arith::MulIOp>(defOp))
    return isStaticallyMultipleOf(mul.getLhs(), k) ||
           isStaticallyMultipleOf(mul.getRhs(), k);
  if (auto add = dyn_cast<arith::AddIOp>(defOp))
    return isStaticallyMultipleOf(add.getLhs(), k) &&
           isStaticallyMultipleOf(add.getRhs(), k);
  if (auto cast = dyn_cast<arith::IndexCastOp>(defOp))
    return isStaticallyMultipleOf(cast.getIn(), k);
  return false;
}

//===----------------------------------------------------------------------===//
// Match Region.
//===----------------------------------------------------------------------===//
//
// Match each per-iteration rc in the body and bin it as a tiled input
// (consumed by `to_tensor`) or tiled output (consumed by
// `materialize_in_destination`). On any local rejection we fail the whole
// match and leave the loop alone.
static std::optional<RegionInfo> matchRegion(scf::ForOp forOp) {
  if (forOp->getParentOfType<scf::ForOp>())
    return std::nullopt;

  // Reject if pre-existing iter_args are present.
  if (!forOp.getInitArgs().empty())
    return std::nullopt;

  RegionInfo info;
  Value iv = forOp.getInductionVar();

  for (Operation &op : forOp.getBody()->getOperations()) {
    auto rcOp = dyn_cast<memref::ReinterpretCastOp>(&op);
    if (!rcOp)
      continue;

    if (!rcOp->hasOneUse())
      return std::nullopt;
    Operation *user = *rcOp->getUsers().begin();

    ArrayRef<int64_t> staticSizes = rcOp.getStaticSizes();
    unsigned tileRank = staticSizes.size();
    if (tileRank != 1 && tileRank != 2) {
      LLVM_DEBUG(DBGS() << "  rc rank not 1 or 2 — rejecting\n");
      return std::nullopt;
    }
    if (ShapedType::isDynamic(staticSizes.back())) {
      LLVM_DEBUG(DBGS() << "  rc innermost size dynamic — rejecting\n");
      return std::nullopt;
    }
    int64_t expectedRowStride = staticSizes.back();
    // rank-1: tile is one "row" per iteration (factor = 1, trivially OK);
    // rank-2 [T, C]: iv must advance by a multiple of T.
    int64_t tileRows = (tileRank == 2) ? staticSizes[0] : 1;

    // lb and step must be statically divisible by tileRows — else
    // iv-as-row-index produces overlapping or skipping tiles.
    if (!isStaticallyMultipleOf(forOp.getLowerBound(), tileRows) ||
        !isStaticallyMultipleOf(forOp.getStep(), tileRows)) {
      LLVM_DEBUG(DBGS() << "  loop lb/step not a multiple of tileRows — "
                           "rejecting\n");
      return std::nullopt;
    }

    auto maybeOffset = validateRcOffset(rcOp, iv, expectedRowStride);
    if (!maybeOffset)
      return std::nullopt;
    Value flatOffset = *maybeOffset;

    if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
      if (tt.getBuffer() != rcOp.getResult())
        return std::nullopt;
      if (!tt.getRestrict())
        return std::nullopt;
      info.inputs.push_back({rcOp, tt, flatOffset});
    } else if (auto mat =
                   dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
      if (mat.getDest() != rcOp.getResult())
        return std::nullopt;
      if (!mat.getRestrict())
        return std::nullopt;
      info.outputs.push_back({rcOp, mat, flatOffset});
    } else {
      return std::nullopt;
    }
  }

  if (info.inputs.empty() || info.outputs.empty())
    return std::nullopt;

  // Disjointness: every rc must have a distinct source memref. If two rcs
  // shared a source, the full-DDR to_tensors we hoist would back the same
  // underlying buffer — making the `restrict` we re-emit on them a lie.
  llvm::SmallPtrSet<Value, 8> sources;
  auto registerSource = [&](Value v) { return sources.insert(v).second; };
  for (InputBinding &in : info.inputs)
    if (!registerSource(in.rc.getSource()))
      return std::nullopt;
  for (OutputBinding &out : info.outputs)
    if (!registerSource(out.rc.getSource()))
      return std::nullopt;

  return info;
}

// Compute the tile/full tensor and memref types for one rc binding into `out`.
// Bails out if the rc isn't rank-1 or rank-2, or has a dynamic inner column.
// fullMemRefType uses offset:? since the hoisted base is a runtime SSA value.
static LogicalResult deriveBindingShape(memref::ReinterpretCastOp rc,
                                        int64_t totalRows, BindingShape &out) {
  ArrayRef<int64_t> tileShape = rc.getStaticSizes();
  unsigned tileRank = tileShape.size();
  if (tileRank != 1 && tileRank != 2)
    return failure();
  if (ShapedType::isDynamic(tileShape.back()))
    return failure();

  auto rcResultType = cast<MemRefType>(rc.getResult().getType());
  Type elemType = rcResultType.getElementType();

  int64_t tileCols = tileShape.back();
  int64_t tileRows = (tileRank == 2) ? tileShape[0] : 1;

  out.tileRows = tileRows;
  out.tileCols = tileCols;
  out.elementType = elemType;
  out.tileTensorType =
      (tileRank == 2) ? RankedTensorType::get({tileRows, tileCols}, elemType)
                      : RankedTensorType::get({tileCols}, elemType);
  out.fullTensorType = RankedTensorType::get({totalRows, tileCols}, elemType);
  out.fullMemRefType = MemRefType::get(
      {totalRows, tileCols}, elemType,
      StridedLayoutAttr::get(elemType.getContext(),
                             /*offset=*/ShapedType::kDynamic, {tileCols, 1}));
  return success();
}

// For each operand of `op` defined by a `tensor.empty` in `emptyOps`, emit a
// fresh `alloc_tensor() {ms=1}` at the current insertion point and remap the
// empty's result to that alloc. Called per-op so each user of the same empty
// gets its own buffer (avoids bufferizer RaW conflict copies). De-dupes
// repeated operand uses within a single op via `seen`.
static void
rewireEmptyOperandsToVtcm(IRRewriter &rewriter, Location loc, Operation *op,
                          const llvm::SmallPtrSet<Operation *, 8> &emptyOps,
                          IRMapping &mapping, IntegerAttr vtcmMs) {

  llvm::SmallPtrSet<Operation *, 4> seen;
  for (Value operand : op->getOperands()) {
    auto emptyOp = operand.getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp || !emptyOps.contains(emptyOp))
      continue;

    // emit exactly one fresh alloc per unique empty per op.
    if (!seen.insert(emptyOp).second)
      continue;

    auto vtcmScratch = bufferization::AllocTensorOp::create(
        rewriter, loc, cast<RankedTensorType>(emptyOp.getResult().getType()),
        emptyOp.getDynamicSizes(), /*copy=*/Value{}, vtcmMs);
    mapping.map(emptyOp.getResult(), vtcmScratch.getResult());
  }
}

//===----------------------------------------------------------------------===//
// Rewrite scf.for : emit extract_slice / insert_slice loop
//===----------------------------------------------------------------------===//
static LogicalResult transformForOp(IRRewriter &rewriter, scf::ForOp forOp,
                                    RegionInfo &info) {
  Location loc = forOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // When UB is constant the hoisted view gets a fully-static shape;
  // otherwise fall back to runtime-sized shape and pipe the SSA value through.
  int64_t totalRows = ShapedType::kDynamic;
  if (auto maybeUB = getConstantValue(forOp.getUpperBound()))
    totalRows = *maybeUB;

  // Compute tile/full tensor and memref types for every binding once,
  // before mutating IR; bail if any binding can't be shaped.
  SmallVector<BindingShape> inShapes(info.inputs.size());
  SmallVector<BindingShape> outShapes(info.outputs.size());
  for (auto [i, in] : llvm::enumerate(info.inputs))
    if (failed(deriveBindingShape(in.rc, totalRows, inShapes[i])))
      return failure();
  for (auto [j, out] : llvm::enumerate(info.outputs))
    if (failed(deriveBindingShape(out.rc, totalRows, outShapes[j])))
      return failure();

  rewriter.setInsertionPoint(forOp);
  auto toIndex = [&](Value v) -> Value {
    if (v.getType().isIndex())
      return v;
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      v);
  };
  Value lbIdx = toIndex(forOp.getLowerBound());
  Value ubIdx = toIndex(forOp.getUpperBound());
  Value stepIdx = toIndex(forOp.getStep());
  Value ubDynamic = (totalRows == ShapedType::kDynamic) ? ubIdx : Value{};

  // Materialize each rc's loop-invariant `base` by cloning the offset SSA
  // chain into the function preamble with iv → 0 substituted. Two rcs that
  // share the same offset chain end up sharing a single hoisted base.
  Value zeroIv = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(forOp.getInductionVar().getType()));
  IRMapping baseMapping;
  baseMapping.map(forOp.getInductionVar(), zeroIv);

  std::function<Value(Value)> cloneBase = [&](Value v) -> Value {
    if (Value mapped = baseMapping.lookupOrNull(v))
      return mapped;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return v;
    if (!forOp->isAncestor(defOp))
      return v;
    for (Value operand : defOp->getOperands())
      cloneBase(operand);
    Operation *cloned = rewriter.clone(*defOp, baseMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      baseMapping.map(oldRes, newRes);
    return baseMapping.lookup(v);
  };

  // helper to create `memref.reinterpret_cast` for full DDR buffer.
  auto buildHoistedRc =
      [&](memref::ReinterpretCastOp orig, Value flatOffset,
          const BindingShape &shape) -> memref::ReinterpretCastOp {
    SmallVector<OpFoldResult> sizes, strides;
    if (totalRows != ShapedType::kDynamic)
      sizes = {rewriter.getIndexAttr(totalRows),
               rewriter.getIndexAttr(shape.tileCols)};
    else
      sizes = {ubDynamic, rewriter.getIndexAttr(shape.tileCols)};
    strides = {rewriter.getIndexAttr(shape.tileCols), rewriter.getIndexAttr(1)};
    Value base = cloneBase(flatOffset);
    return memref::ReinterpretCastOp::create(
        rewriter, loc, shape.fullMemRefType, orig.getSource(), base, sizes,
        strides);
  };

  // Create reinterpret_cast and `bufferization.to_tensor` for the full
  // buffer. Output full tensor needs to become init to scf.for.
  SmallVector<memref::ReinterpretCastOp> newInRcs(info.inputs.size());
  SmallVector<memref::ReinterpretCastOp> newOutRcs(info.outputs.size());

  for (auto [i, in] : llvm::enumerate(info.inputs))
    newInRcs[i] = buildHoistedRc(in.rc, in.flatOffset, inShapes[i]);

  for (auto [j, out] : llvm::enumerate(info.outputs))
    newOutRcs[j] = buildHoistedRc(out.rc, out.flatOffset, outShapes[j]);

  SmallVector<Value> inFullTensors(info.inputs.size());
  for (size_t i = 0, e = info.inputs.size(); i < e; ++i) {
    auto tt = bufferization::ToTensorOp::create(
        rewriter, loc << "input_tensor_ddr", inShapes[i].fullTensorType,
        newInRcs[i].getResult(),
        /*restrict=*/true, /*writable=*/false);
    inFullTensors[i] = tt.getResult();
  }

  SmallVector<Value> outInits(info.outputs.size());
  for (size_t j = 0, e = info.outputs.size(); j < e; ++j) {
    auto tt = bufferization::ToTensorOp::create(
        rewriter, loc << "output_tensor_ddr", outShapes[j].fullTensorType,
        newOutRcs[j].getResult(),
        /*restrict=*/true, /*writable=*/true);
    outInits[j] = tt.getResult();
  }

  SmallVector<OpFoldResult> sharedStrides = {rewriter.getIndexAttr(1),
                                             rewriter.getIndexAttr(1)};

  // Now create the new scf.for.
  auto newForOp = scf::ForOp::create(
      rewriter, loc, lbIdx, ubIdx, stepIdx, ValueRange(outInits),
      [](OpBuilder &, Location, Value, ValueRange) {});

  rewriter.setInsertionPointToEnd(newForOp.getBody());
  Value newIv = newForOp.getInductionVar();
  ValueRange iterArgs = newForOp.getRegionIterArgs();

  // VTCM memory space attribute (memory_space = 1 on alloc_tensor).
  IntegerAttr vtcmMs = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1);

  IRMapping mapping;
  // If the original IV was integer-typed insert a cast back so that ops
  // using the original IV directly (e.g. arith.muli %iv, %cst : i32)
  // get a correctly-typed replacement.
  if (!forOp.getInductionVar().getType().isIndex()) {
    Value ivCast = arith::IndexCastOp::create(
        rewriter, loc, forOp.getInductionVar().getType(), newIv);
    mapping.map(forOp.getInductionVar(), ivCast);
  } else {
    mapping.map(forOp.getInductionVar(), newIv);
  }

  // Tiled inputs: extract_slice [iv, 0] [tileRows, tileCols] [1, 1] →
  // alloc_tensor(copy, ms=1). Result type = tileTensorType, which is rank-1
  // (rank-reducing slice) or rank-2 (preserved) depending on the original tile.
  for (auto [i, in] : llvm::enumerate(info.inputs)) {
    SmallVector<OpFoldResult> offsets = {newIv, rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(inShapes[i].tileRows),
        rewriter.getIndexAttr(inShapes[i].tileCols)};
    auto extractSlice = tensor::ExtractSliceOp::create(
        rewriter, loc << "input_slice_ddr", inShapes[i].tileTensorType,
        inFullTensors[i], offsets, sizes, sharedStrides);
    auto vtcmIn = bufferization::AllocTensorOp::create(
        rewriter, loc << "input_slice_vtcm", inShapes[i].tileTensorType,
        /*dynamicSizes=*/ValueRange{}, extractSlice.getResult(), vtcmMs);
    mapping.map(in.toTensor.getResult(), vtcmIn.getResult());
  }

  llvm::SmallPtrSet<Operation *, 16> replacedOps;
  for (InputBinding &in : info.inputs) {
    replacedOps.insert(in.rc.getOperation());
    replacedOps.insert(in.toTensor.getOperation());
  }
  for (OutputBinding &out : info.outputs) {
    replacedOps.insert(out.rc.getOperation());
    replacedOps.insert(out.materialize.getOperation());
  }

  // tensor.empty operands — defined inside the body — get fresh per-use
  // alloc_tensor(ms=1) replacements. Mapping each empty 1:1 to a single
  // alloc would force OneShotBufferize to emit conflict-resolution copies in
  // DDR (TCM→DDR→TCM roundtrips) when the buffer is overwritten while a
  // previous value is still live. Per-use allocs give every writer its own
  // buffer.
  llvm::SmallPtrSet<Operation *, 8> emptyOps;
  for (Operation &op : forOp.getBody()->without_terminator())
    if (isa<tensor::EmptyOp>(&op))
      emptyOps.insert(&op);
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (replacedOps.contains(&op))
      continue;
    if (auto cast = dyn_cast<arith::IndexCastOp>(&op)) {
      if (cast.getIn() == forOp.getInductionVar()) {
        mapping.map(cast.getResult(), newIv);
        continue;
      }
    }
    if (isa<tensor::EmptyOp>(&op))
      continue;
    rewireEmptyOperandsToVtcm(rewriter, loc, &op, emptyOps, mapping, vtcmMs);
    rewriter.clone(op, mapping);
  }

  // Generate the sequence:
  //   %tensor = alloc_tensor copy(%mapping) {memory_space = 1 : i64}
  //   %inserted_slice = tensor.insert_slice %tensor into %...
  //   yield %inserted_slice, ...
  // Note : explicit alloc_tensor(copy=srcTensor, ms=1) forces the
  // bufferizer to keep the compute result in a separate VTCM buffer.
  // Without it, the result would alias with the DDR output subview directly,
  // and appear before the input subview→alloc→copy triplets in the bufferized
  // body and therefore break tiled-generic expected ordering.
  SmallVector<Value> yields(info.outputs.size());
  for (auto [j, out] : llvm::enumerate(info.outputs)) {
    Value srcTensor = mapping.lookupOrDefault(out.materialize.getSource());
    SmallVector<OpFoldResult> offsets = {newIv, rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(outShapes[j].tileRows),
        rewriter.getIndexAttr(outShapes[j].tileCols)};
    auto vtcmOut = bufferization::AllocTensorOp::create(
        rewriter, loc << "output_slice_vtcm", outShapes[j].tileTensorType,
        /*dynamicSizes=*/ValueRange{}, srcTensor, vtcmMs);
    auto inserted = tensor::InsertSliceOp::create(
        rewriter, loc << "output_slice_ddr", vtcmOut.getResult(), iterArgs[j],
        offsets, sizes, sharedStrides);
    yields[j] = inserted.getResult();
  }

  // Add yields and the tags of tiled_generic and parallel.
  scf::YieldOp::create(rewriter, loc, yields);
  newForOp->setAttr("tiled_generic", UnitAttr::get(ctx));
  newForOp->setAttr("all_parallel", UnitAttr::get(ctx));

  // materialize writes outside the scf.for which used to happen inside.
  rewriter.setInsertionPointAfter(newForOp);
  for (size_t j = 0, e = info.outputs.size(); j < e; ++j) {
    auto finalMat = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, TypeRange{}, newForOp.getResult(j),
        newOutRcs[j].getResult());
    finalMat.setRestrict(true);
    finalMat.setWritable(true);
  }

  rewriter.eraseOp(forOp);
  return success();
}

//===----------------------------------------------------------------------===//
// PromoteTiledSCFLoopToVTCMPass
//===----------------------------------------------------------------------===//
struct PromoteTiledSCFLoopToVTCMPass
    : public ::impl::PromoteTiledSCFLoopToVTCMBase<
          PromoteTiledSCFLoopToVTCMPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<scf::ForOp> candidates;
    funcOp.walk([&](scf::ForOp forOp) { candidates.push_back(forOp); });

    for (scf::ForOp forOp : candidates) {
      auto matchFound = matchRegion(forOp);
      if (!matchFound)
        continue;
      if (failed(transformForOp(rewriter, forOp, *matchFound)))
        LLVM_DEBUG(DBGS() << "transformation failed on candidate\n");
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createPromoteTiledSCFLoopToVTCMPass() {
  return std::make_unique<PromoteTiledSCFLoopToVTCMPass>();
}
