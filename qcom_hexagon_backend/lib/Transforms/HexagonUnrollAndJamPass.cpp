//===- HexagonUnrollAndJamPass.cpp - Outer-loop unroll-and-jam ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass unroll-and-jams the outermost qualifying scf.for loop in each loop
// nest. Where scf-loop-unroll targets innermost loops and just clones the body,
// unroll-and-jam scales the outer step by the factor and interleaves the copies
// of the body. When the body contains a reduction, each unrolled iteration is
// an independent reduction chain, so the in-order VLIW scheduler can overlap
// their latencies -> more ILP.
//
// Two jam mechanics, selected by body shape:
//
//   * Real loop *nest* (the body has an inner scf.for): delegated to upstream
//     mlir::loopUnrollJamByFactor, which fuses the duplicated inner loops --
//     the classic jam.
//
//   * Straight-line body (the reduction kernels we target -- a sequence of
//     linalg ops with no inner loop): upstream has no inner loop to fuse, so it
//     can only *append* the copies: A1 A2 .. An B1 B2 .. Bn. Inside a row the
//     ops form a dependency chain, so appending leaves every independent op
//     (the B's) far from the latency bubble it could fill -> no ILP, and on
//     kernels whose body holds an opaque call (e.g. rmsnorm's sqrt) the
//     downstream LLVM scheduler cannot recover it either. So here we interleave
//     the copies ourselves, round-robin by position: A1 B1 A2 B2 .. An Bn. This
//     reproduces the ordering of the hand-written 2-rows-per-iteration kernel.
//
// Upstream loopUnrollJamByFactor synthesizes its scaled step and shifted
// induction variable as index-typed constants, so before jamming we normalize
// an integer-typed loop induction variable to index (re-casting to the original
// type inside the body). This lets triton-shared's i32 row loops be jammed
// without hitting an arith.muli(i32, index) type mismatch.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexagon-unroll-and-jam"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONUNROLLANDJAM
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Extract a constant integer value from a Value (bounds/step are typically
/// arith.constant here). Mirrors the helper in SCFLoopUnrollPass.
static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

/// Returns the constant trip count of `forOp` if its bounds and step are all
/// constant and the loop makes forward progress, otherwise std::nullopt.
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return std::nullopt;

  int64_t diff = *ub - *lb;
  if (diff <= 0)
    return std::nullopt;

  // Equivalent to ceil((ub - lb) / step).
  return 1 + ((diff - 1) / *step);
}

/// Returns true if the control operands (lb/ub/step) of every inner scf.for are
/// defined outside `forOp`. This mirrors the precondition the upstream utility
/// checks; we replicate it so we can reject early, before mutating any IR.
static bool areInnerBoundsInvariant(scf::ForOp forOp) {
  auto walkResult = forOp.walk([&](scf::ForOp innerForOp) {
    if (innerForOp == forOp)
      return WalkResult::advance();
    if (!forOp.isDefinedOutsideOfLoop(innerForOp.getLowerBound()) ||
        !forOp.isDefinedOutsideOfLoop(innerForOp.getUpperBound()) ||
        !forOp.isDefinedOutsideOfLoop(innerForOp.getStep()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Finds the first reduction-carrying LinalgOp in the loop body, if any (null
/// LinalgOp otherwise). Both linalg.reduce and a reduction-carrying
/// linalg.generic implement the LinalgOp interface and report reduction
/// iterators, so one interface query covers RMSNorm/LayerNorm/Softmax without
/// hardcoding op names. A pure elementwise loop has no serial chain to overlap,
/// so jamming it would just add register pressure, so callers reject a null
/// result.
static linalg::LinalgOp findReductionOp(scf::ForOp forOp) {
  linalg::LinalgOp result;
  forOp.getBody()->walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp.getNumReductionLoops() > 0) {
      result = linalgOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Reduction row length C: the number of elements a reduction sweeps per row.
/// Computed as the max, over every reduction-carrying LinalgOp in the body, of
/// the product of that op's static iteration space. The max (rather than the
/// first op's) is robust to SplitReduceGeneric having reshaped one reduce over
/// C into a [parallel x reduce] partial-reduce (whose iteration product is
/// still C) plus a small final reduce -- we want the C-sized one regardless of
/// IR order. Reduction ops with a dynamic iteration dim are skipped -- their
/// length is unknown and can't inform the cap; a dynamic sibling must not
/// suppress a known-over-cap static reduce. Returns nullopt only when no
/// reduction has a statically-known length.
static std::optional<int64_t> reductionRowLength(scf::ForOp forOp) {
  std::optional<int64_t> maxLen;
  forOp.getBody()->walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp.getNumReductionLoops() == 0)
      return; // no reduction loop -> skip this op
    int64_t product = 1;
    for (int64_t dim : linalgOp.getStaticLoopRanges()) {
      if (ShapedType::isDynamic(dim))
        return; // unknown length -> this op can't inform the cap, skip it
      product *= dim;
    }
    if (!maxLen || product > *maxLen)
      maxLen = product;
  });
  return maxLen;
}

/// Total op count in the loop body (nested ops included). The jammed loop will
/// hold `factor` copies of this, so we cap `bodyOps * factor` to keep the
/// duplicated body from spilling. This counts *distinct ops* and so is
/// orthogonal to `reductionRowLength` (the C gate):
/// this bounds how many ops we duplicate, C bounds how long each reduction is.
static int64_t computeBodyLength(scf::ForOp forOp) {
  int64_t count = 0;
  forOp.getBody()->walk([&](Operation *op) { ++count; });
  return count;
}

/// The ONE place profitability/legality lives. Returns the factor to jam by, or
/// 0 to mean "do not jam".
static int64_t chooseUnrollJamFactor(scf::ForOp forOp, int64_t factor,
                                     int64_t minTripCount, int64_t maxBodyOps,
                                     int64_t maxReductionRowLength) {
  // ---- hard correctness gates (mirror loopUnrollJamByFactor, reject early) --
  if (factor <= 1) {
    DBG("factor <= 1, nothing to do");
    return 0;
  }
  // The underlying MLIR utility does not support loops with results/iter_args.
  // Note that this isn't a real limitation for us, because the reduction
  // kernel we target (rmsnorm/layernorm/softmax) accumulate into memrefs via
  // linalg.reduce and memref.store so the row loop carries no loop-carried SSA
  // value
  if (forOp->getNumResults() > 0 || !forOp.getInitArgs().empty()) {
    DBG("loop has results/iter_args, skipping");
    return 0;
  }
  if (!areInnerBoundsInvariant(forOp)) {
    DBG("inner loop bounds are not invariant, skipping");
    return 0;
  }
  std::optional<int64_t> tripCount = getConstantTripCount(forOp);
  if (!tripCount) {
    DBG("non-constant trip count, skipping");
    return 0;
  }
  if (*tripCount % factor != 0) {
    // Note this condition also rejects *tripCount < factor at the same time
    DBG("trip count " << *tripCount << " not a multiple of factor " << factor);
    return 0;
  }

  // ---- profitability (tunable) ----------------------------------------------
  if (*tripCount < minTripCount) {
    DBG("trip count " << *tripCount << " below min " << minTripCount);
    return 0;
  }
  if (!findReductionOp(forOp)) {
    // A body with no reduction has no long serial dependent chain - its ops are
    // already independent and the pipeline is not latency-bound. Jamming would
    // add register pressure without a stall to hide, so skip it.
    DBG("body has no reduction: no serial chain to overlap, skipping");
    return 0;
  }
  // Shape gate: past this row length the per-row elementwise stream dominates
  // and jamming stops paying off. Only gate when C is statically known.
  std::optional<int64_t> rowLen = reductionRowLength(forOp);
  if (rowLen && *rowLen > maxReductionRowLength) {
    DBG("reduction row length " << *rowLen << " exceeds cap "
                                << maxReductionRowLength << ", skipping");
    return 0;
  }
  // Register-pressure gate: the jammed loop will hold `factor` copies of the
  // body, so cap the duplicated op count to keep it from spilling. Orthogonal
  // to the C gate above -- that bounds each reduction's length, this bounds how
  // many distinct ops we clone.
  int64_t bodyCost = computeBodyLength(forOp);
  if (bodyCost * factor > maxBodyOps) {
    DBG("jammed body would cost " << (bodyCost * factor) << " exceeds cap "
                                  << maxBodyOps);
    return 0;
  }

  DBG("loop qualifies for unroll-and-jam by factor " << factor);
  return factor;
}

/// Normalize an integer-typed loop induction variable to `index`.
///
/// Upstream loopUnrollJamByFactor() synthesizes the scaled step and shifted IV
/// as index-typed constants; if the IV is i32 those would form
/// arith.muli(i32, index) / arith.addi(i32, index) and fail the verifier. We
/// rebuild the loop with index bounds and cast the index IV back to the
/// original integer type at the top of the body, so all existing body uses stay
/// well-typed. ConvertIndexToLLVM (bitwidth 32) later collapses the index math
/// back to the same integer the baseline had.
///
/// Precondition: `forOp` has constant bounds/step and no iter_args (guaranteed
/// by chooseUnrollJamFactor). Returns the potentially modified loop.
static scf::ForOp normalizeIVToIndex(scf::ForOp forOp, IRRewriter &rewriter) {
  Type ivType = forOp.getInductionVar().getType();
  if (isa<IndexType>(ivType))
    return forOp;

  Location loc = forOp.getLoc();
  int64_t lb = *getConstantIntValue(forOp.getLowerBound());
  int64_t ub = *getConstantIntValue(forOp.getUpperBound());
  int64_t step = *getConstantIntValue(forOp.getStep());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);

  Value lbIdx = arith::ConstantIndexOp::create(rewriter, loc, lb);
  Value ubIdx = arith::ConstantIndexOp::create(rewriter, loc, ub);
  Value stepIdx = arith::ConstantIndexOp::create(rewriter, loc, step);

  // No iter_args (gated), so the new loop's body is just an empty scf.yield.
  auto newForOp = scf::ForOp::create(rewriter, loc, lbIdx, ubIdx, stepIdx);

  Block *newBody = newForOp.getBody();
  rewriter.setInsertionPointToStart(newBody);

  // Cast the index IV back to the original integer type for the body uses.
  Value castIV = arith::IndexCastOp::create(rewriter, loc, ivType,
                                            newForOp.getInductionVar());

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), castIV);
  for (Operation &op : forOp.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  forOp->erase();
  DBG("normalized " << ivType << " induction variable to index");
  return newForOp;
}

/// True if `forOp`'s body contains an inner scf.for. The straight-line
/// interleaving path below assumes a flat body; a real loop nest is handed to
/// upstream loopUnrollJamByFactor instead (it fuses the duplicated inner
/// loops).
static bool bodyHasInnerLoop(scf::ForOp forOp) {
  bool found = false;
  forOp.getBody()->walk([&](scf::ForOp inner) {
    if (inner != forOp) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Hook for a future memory-effect safety check.
///
/// Round-robin interleaving reorders row B's ops relative to row A's (e.g. B's
/// loads move ahead of A's stores). That is correct only when the unrolled
/// iterations are data-independent -- the standard unroll-and-jam precondition,
/// which today we assume the caller/heuristic has established (the reduction
/// gate is the current proxy). This is the single place to tighten later:
/// return false to fall back to plain (append) jamming when independence cannot
/// be proven. Kept trivial for now so the safe/unsafe decision has one home.
static bool interleavingIsSafe(scf::ForOp /*forOp*/, int64_t /*factor*/) {
  return true;
}

/// Unroll-and-jam a straight-line (no inner loop) body by `factor`, in place.
///
/// This is our full replacement for upstream loopUnrollJamByFactor on the
/// straight-line path: it does *both* activities itself -- it unrolls (clones
/// the body `factor-1` times) *and* jams them, where for a straight-line body
/// the jam is a round-robin interleave. Upstream is not called here; it is used
/// only for real loop *nests* (see the caller).
///
/// Turns the single-row body  A1 A2 .. An  into the round-robin order
/// A1 B1 .. A2 B2 .. An Bn  for factor 2 (and analogously A1 B1 C1 A2 .. for
/// larger factors), then scales the loop step by `factor`. Copy `c` (c >= 1)
/// has its induction-variable uses remapped to `iv + c*step`, exactly as
/// upstream does, so each copy addresses its own row.
///
/// Correctness: within a copy the original op order is preserved, so every
/// intra-row def-use edge is kept; copies are data-independent (different
/// rows), which `interleavingIsSafe` guards. Emitting by rounds (all copies'
/// position p, then position p+1) keeps SSA valid -- an op at position p only
/// uses values from positions < p of its own copy (or loop-invariants), all
/// already emitted.
///
/// Precondition: constant step, no iter_args, index-typed IV (guaranteed by
/// chooseUnrollJamFactor + normalizeIVToIndex). Body has no inner loop.
static LogicalResult unrollAndInterleaveStraightLineBody(scf::ForOp forOp,
                                                         int64_t factor,
                                                         IRRewriter &rewriter) {
  Block *body = forOp.getBody();
  int64_t step = *getConstantIntValue(forOp.getStep());
  Value iv = forOp.getInductionVar();
  Location loc = forOp.getLoc();

  // The original ops, in order, excluding the terminator. This is copy 0 ("A").
  SmallVector<Operation *> rowA;
  for (Operation &op : body->without_terminator())
    rowA.push_back(&op);
  if (rowA.empty())
    return success(); // Nothing to jam.

  OpBuilder::InsertionGuard guard(rewriter);

  // Build copies 1..factor-1 ("B", "C", ...). The shifted-IV computation for a
  // copy is emitted at the *start* of the body so it dominates every clone that
  // uses it (it depends only on the induction variable, a block argument); the
  // cloned ops go right before the terminator. clones[c][p] is position p of
  // copy c (index c-1 here).
  SmallVector<SmallVector<Operation *>> clones(factor - 1);
  for (int64_t c = 1; c < factor; ++c) {
    IRMapping map;
    if (!iv.use_empty()) {
      // iv' = iv + c*step, matching upstream's per-copy induction shift.
      rewriter.setInsertionPointToStart(body);
      Value tag = arith::ConstantIndexOp::create(rewriter, loc, step * c);
      Value shifted = arith::AddIOp::create(rewriter, loc, iv, tag);
      map.map(iv, shifted);
    }
    rewriter.setInsertionPoint(body->getTerminator());
    clones[c - 1].reserve(rowA.size());
    for (Operation *op : rowA)
      clones[c - 1].push_back(rewriter.clone(*op, map));
  }

  // Round-robin reorder: for each position p, move copy 1..factor-1's op at p
  // to sit right after copy 0's op at p. Copy 0 already holds the correct
  // relative order, so we only slot the clones in behind their A counterpart.
  // Iterating p ascending keeps each A-anchor ahead of the clones inserted
  // after it.
  for (size_t p = 0; p < rowA.size(); ++p) {
    Operation *anchor = rowA[p];
    for (int64_t c = 1; c < factor; ++c) {
      clones[c - 1][p]->moveAfter(anchor);
      anchor = clones[c - 1][p]; // chain: A_p, B_p, C_p, ... stay contiguous
    }
  }

  // Scale the step by the factor (the loop now advances `factor` rows/iter).
  rewriter.setInsertionPoint(forOp);
  Value newStep = arith::ConstantIndexOp::create(rewriter, loc, step * factor);
  forOp.getStepMutable().assign(newStep);

  DBG("interleaved straight-line body by factor " << factor);
  return success();
}

struct HexagonUnrollAndJamPass
    : public ::impl::HexagonUnrollAndJamBase<HexagonUnrollAndJamPass> {

  explicit HexagonUnrollAndJamPass(
      const mlir::hexagon::HexagonUnrollAndJamOptions &options)
      : HexagonUnrollAndJamBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    DBG("Running unroll-and-jam with factor " << unrollJamFactor);

    // S1: jam the OUTERMOST qualifying loop in each nest (opposite of the
    // innermost scf-loop-unroll). Collect loops with no enclosing scf.for.
    SmallVector<scf::ForOp> candidates;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>())
        candidates.push_back(forOp);
    });
    DBG("Found " << candidates.size() << " outermost loops to consider");

    IRRewriter rewriter(context);
    for (scf::ForOp forOp : candidates) {
      // Skip if a previous transform already erased this op.
      if (!forOp->getBlock())
        continue;

      int64_t factor =
          chooseUnrollJamFactor(forOp, unrollJamFactor, minTripCount,
                                maxBodyOps, maxReductionRowLength);
      if (factor <= 1)
        continue;

      // Resolve the i32-IV blocker (Option 3A) before jamming.
      scf::ForOp loop = normalizeIVToIndex(forOp, rewriter);

      // Straight-line body (like our reduction kernels) we can safely reorder:
      // interleave the copies ourselves into A1 B1 A2 B2 .. An Bn, because
      // mlir's utility would only append them (A1..An B1..Bn), which does not
      // create more ILP.
      if (!bodyHasInnerLoop(loop) && interleavingIsSafe(loop, factor)) {
        if (failed(unrollAndInterleaveStraightLineBody(loop, factor, rewriter)))
          DBG("unrollAndInterleaveStraightLineBody failed unexpectedly, "
              "skipping");
        else
          DBG("Successfully unroll-and-interleaved loop");
        continue; // Continue with the next candidate
      }

      // Everything else -- a real loop nest, or a straight-line body whose
      // independence we could not establish -- hands off to upstream
      // loopUnrollJamByFactor: it fuses the duplicated inner loops of a nest,
      // and for a straight-line body falls back to appending the copies.
      if (failed(loopUnrollJamByFactor(loop, factor)))
        // The predicate should have prevented an outright failure; log and
        // leave valid IR rather than failing the whole compile.
        DBG("loopUnrollJamByFactor failed, skipping");
      else
        DBG("Successfully unroll-and-jammed loop");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createHexagonUnrollAndJamPass(
    const HexagonUnrollAndJamOptions &options) {
  return std::make_unique<HexagonUnrollAndJamPass>(options);
}
