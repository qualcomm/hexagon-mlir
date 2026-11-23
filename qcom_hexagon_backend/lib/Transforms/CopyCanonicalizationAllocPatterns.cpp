//===- CopyCanonicalizationAllocPatterns.cpp ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Alloc-anchored copy-canonicalization patterns:
//   * `RedirectAllocWritesToDestinationPattern` — redirect writers of a
//     write-only staging `memref.alloc` to its drain destination(s).
//   * `EliminateAllocInIfWithCopyPattern` — eliminate a write-only alloc
//     yielded from an `scf.if` then-branch and copied to the else-yielded
//     destination.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/CopyCanonicalizationAllocPatterns.h"
#include "hexagon/Transforms/CopyCanonicalizationCommon.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::hexagon::detail;

#define DEBUG_TYPE "copy-canonicalization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using ClaimMap = llvm::SmallDenseMap<Operation *, SmallVector<OpOperand *>, 2>;

/// Hexagon convention: memory space 0 (no attribute) is DDR, memory space 1
/// is VTCM.  Non-integer attrs are treated as DDR.
static int64_t memorySpaceAsInt(Attribute attr) {
  if (auto i = dyn_cast_or_null<IntegerAttr>(attr))
    return i.getInt();
  return 0;
}

/// Returns true when the types have the same shape, element type, and static
/// strides. Offsets and memory spaces may differ because cloned view ops
/// re-infer their result types from the new source.
/// Dynamic strides are rejected because equal dynamic markers do not imply
/// equal runtime strides.
static bool canRetypeViaShape(MemRefType allocType, MemRefType destType) {
  if (allocType.getShape() != destType.getShape() ||
      allocType.getElementType() != destType.getElementType())
    return false;
  int64_t allocOffset, destOffset;
  SmallVector<int64_t> allocStrides, destStrides;
  if (failed(allocType.getStridesAndOffset(allocStrides, allocOffset)) ||
      failed(destType.getStridesAndOffset(destStrides, destOffset)))
    return allocType.getLayout() == destType.getLayout();
  if (llvm::any_of(allocStrides, ShapedType::isDynamic) ||
      llvm::any_of(destStrides, ShapedType::isDynamic))
    return false;
  // Strides must match; the offset is allowed to differ and is carried by the
  // re-inferred view types (see `redirectWriteOnlyUses`).
  return allocStrides == destStrides;
}

/// Returns true when replacing `allocType` with `destType` would lose an
/// offset or non-strided layout difference in existing view result types.
static bool offsetWouldBeDropped(MemRefType allocType, MemRefType destType) {
  int64_t allocOffset, destOffset;
  SmallVector<int64_t> allocStrides, destStrides;
  if (failed(allocType.getStridesAndOffset(allocStrides, allocOffset)) ||
      failed(destType.getStridesAndOffset(destStrides, destOffset)))
    // Non-strided layout difference: only safe when the layouts are identical.
    return allocType.getLayout() != destType.getLayout();
  return allocOffset != destOffset;
}

/// Among `drains` (sibling copies of an alloc), return the one that
/// "claims" `writer`: a drain whose block contains an ancestor of `writer`
/// that lies before the drain.  When several qualify, pick the earliest in
/// program order via `happensBefore` (handles cross-block cases).
static memref::CopyOp findClaimingDrain(Operation *writer,
                                        ArrayRef<memref::CopyOp> drains) {
  memref::CopyOp claim;
  for (auto drain : drains) {
    Operation *ancestor = drain->getBlock()->findAncestorOpInBlock(*writer);
    if (!ancestor || !ancestor->isBeforeInBlock(drain.getOperation()))
      continue;
    if (!claim || happensBefore(drain.getOperation(), claim.getOperation()))
      claim = drain;
  }
  return claim;
}

/// Redirect every operand in `uses` to `newRoot`. All operands must share
/// the same current value (the alloc result, or a view's result during
/// recursion). For `ViewLikeOpInterface` users, the op is cloned with the
/// source substituted and the result type updated to propagate `newRoot`'s
/// memory space and leading offset. `memref.subview` result types are
/// re-inferred; other views only re-stamp the memory space. We then recurse
/// on every use of the original view's result. The caller must have validated
/// the chain via `isWriteOnlyUseChain`.
static void redirectWriteOnlyUses(PatternRewriter &rewriter,
                                  ArrayRef<OpOperand *> uses, Value newRoot) {
  for (OpOperand *use : uses) {
    Operation *user = use->getOwner();
    if (isa<ViewLikeOpInterface>(user)) {
      assert(user->getNumResults() == 1 && "expected single-result view op");
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(user);
      Operation *cloned = rewriter.clone(*user);
      Value oldRoot = use->get();
      for (OpOperand &operand : cloned->getOpOperands())
        if (operand.get() == oldRoot)
          operand.set(newRoot);
      Value newResult = cloned->getResult(0);
      auto oldResultType = cast<MemRefType>(newResult.getType());
      auto newRootType = cast<MemRefType>(newRoot.getType());
      if (auto subview = dyn_cast<memref::SubViewOp>(cloned)) {
        // Re-infer from the new source to propagate its layout. Preserve the
        // existing result shape because the subview may be rank-reduced.
        newResult.setType(memref::SubViewOp::inferRankReducedResultType(
            oldResultType.getShape(), newRootType, subview.getStaticOffsets(),
            subview.getStaticSizes(), subview.getStaticStrides()));
      } else {
        // Non-subview view: preserve shape/layout, track the new memory space.
        newResult.setType(MemRefType::get(
            oldResultType.getShape(), oldResultType.getElementType(),
            oldResultType.getLayout(), newRootType.getMemorySpace()));
      }
      SmallVector<OpOperand *, 4> nestedUses;
      for (OpOperand &nested : user->getResult(0).getUses())
        nestedUses.push_back(&nested);
      redirectWriteOnlyUses(rewriter, nestedUses, newResult);
      rewriter.eraseOp(user);
      continue;
    }
    rewriter.modifyOpInPlace(user, [&]() { use->set(newRoot); });
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

/// Eliminate a write-only staging `memref.alloc` and the `memref.copy`s
/// that drain it into one or more destinations.  Each writer of the alloc
/// (or of an aliasing view) is redirected to the destination of the next
/// drain that follows it in execution order.
///
/// Anchored on `memref::AllocOp` so a single match invocation sees the
/// whole drain pattern at once.
///
/// Single-drain example:
///   ```
///   %alloc = memref.alloc() : memref<128xf32>
///   scf.for %tile = 0 to 128 step 32 {
///     %sv = memref.subview %alloc[%tile][32][1] : ...
///     vector.transfer_write %vec, %sv[%c0] : ...
///   }
///   memref.copy %alloc, %dst : memref<128xf32>
///   ```
/// becomes:
///   ```
///   scf.for %tile = 0 to 128 step 32 {
///     %sv = memref.subview %dst[%tile][32][1] : ...
///     vector.transfer_write %vec, %sv[%c0] : ...
///   }
///   ```
///
/// Multi-drain example:
///   ```
///   %alloc = memref.alloc()
///   write %a, %alloc
///   memref.copy %alloc, %dst1
///   write %b, %alloc
///   memref.copy %alloc, %dst2
///   ```
/// becomes:
///   ```
///   write %a, %dst1
///   write %b, %dst2
///   ```
struct RedirectAllocWritesToDestinationPattern final
    : public OpRewritePattern<memref::AllocOp> {
  RedirectAllocWritesToDestinationPattern(MLIRContext *ctx,
                                          DominanceInfo &domInfo)
      : OpRewritePattern(ctx), domInfo(domInfo) {}

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Step 1 — Categorize alloc users into draining copies and write-only
    // chain roots.  Any other use disqualifies the rewrite.
    SmallVector<memref::CopyOp> drains;
    SmallVector<OpOperand *> writerUses;
    for (OpOperand &use : allocOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (auto cp = dyn_cast<memref::CopyOp>(user))
        if (cp.getSource() == allocOp.getResult()) {
          drains.push_back(cp);
          continue;
        }
      if (!isWriteOnlyUseRoot(use, /*consumer=*/nullptr))
        return failure();
      writerUses.push_back(&use);
    }
    if (drains.empty() || writerUses.empty())
      return failure();

    // Step 2 — Pair each writer with its claiming drain.  Bail when any
    // writer has no claim (alloc not fully consumed by drains we can
    // rewrite).
    ClaimMap claims;
    for (OpOperand *use : writerUses) {
      memref::CopyOp drain = findClaimingDrain(use->getOwner(), drains);
      if (!drain)
        return failure();
      claims[drain.getOperation()].push_back(use);
    }

    // Step 3 — Split drains by claim status, preserving input order.
    SmallVector<memref::CopyOp> claimedDrains, unclaimedDrains;
    for (auto drain : drains)
      (claims.count(drain.getOperation()) ? claimedDrains : unclaimedDrains)
          .push_back(drain);

    // Step 4 — Memory-hierarchy guard.  Bail when an alloc in a higher
    // memory space (e.g. VTCM) drains into a lower one (e.g. DDR);
    // redirecting the writes would lose the staging benefit of
    // "stage hot, then bulk copy".
    auto allocType = allocOp.getType();
    int64_t allocSpace = memorySpaceAsInt(allocType.getMemorySpace());
    for (auto drain : claimedDrains) {
      auto destType = cast<MemRefType>(drain.getTarget().getType());
      if (allocSpace > memorySpaceAsInt(destType.getMemorySpace()))
        return failure();
    }

    // Step 5 — Coverage guard.  After the rewrite, claimed drains are
    // erased and the alloc has no writers.  Any remaining unclaimed drain
    // would read an empty alloc — bail when one originally observed
    // writer content via `happensBefore` (handles same-block and
    // cross-block reachability).  Drains that were already reading
    // uninitialized content are tolerated.
    for (auto drain : unclaimedDrains)
      for (OpOperand *use : writerUses)
        if (happensBefore(use->getOwner(), drain.getOperation()))
          return failure();

    // Step 6 — Validate dominance and cast feasibility before mutating.
    // Writer-dominance is required unconditionally; the alloc-dominance and
    // cast-feasibility checks only matter on the cast path (a cast inserted
    // at allocOp must be reachable from `dest`).
    for (auto drain : claimedDrains) {
      Value dest = drain.getTarget();
      auto destType = cast<MemRefType>(dest.getType());
      for (OpOperand *use : claims.lookup(drain.getOperation()))
        if (!domInfo.properlyDominates(dest, use->getOwner()))
          return failure();
      if (!canRetypeViaShape(allocType, destType)) {
        if (!domInfo.properlyDominates(dest, allocOp.getOperation()))
          return failure();
        if (!canBuildDestReplacement(dest, allocType))
          return failure();
      }
    }

    // Step 7 — Apply.  For each claimed drain, build the replacement root
    // (dest itself or a cast chain) and redirect its claimed writer-uses;
    // erase the drain.  Erase the alloc once nothing references it.
    for (auto drain : claimedDrains) {
      Value dest = drain.getTarget();
      auto destType = cast<MemRefType>(dest.getType());
      Value newRoot =
          canRetypeViaShape(allocType, destType)
              ? dest
              : buildDestReplacement(rewriter, dest, allocType, allocOp);
      assert(newRoot && "Step 6 must have proven the cast chain feasible");
      redirectWriteOnlyUses(rewriter, claims.lookup(drain.getOperation()),
                            newRoot);
      rewriter.eraseOp(drain);
    }
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);

    DBG("RedirectAllocWritesToDestination: redirected "
        << writerUses.size() << " writer(s) across " << claimedDrains.size()
        << " drain(s)");
    return success();
  }

private:
  DominanceInfo &domInfo;
};

/// Eliminate a write-only `memref.alloc` defined in an `scf.if` then-branch
/// that is yielded and then copied to a destination — redirect the
/// then-branch writes to write directly to that destination.
///
/// Matches:
///   ```
///   %result = scf.if %cond {
///     %alloc = memref.alloc()      // <-- match anchor
///     ... writes to %alloc ...
///     scf.yield %alloc
///   } else {
///     scf.yield %dest
///   }
///   memref.copy %result, %dest
///   ```
struct EliminateAllocInIfWithCopyPattern final
    : public OpRewritePattern<memref::AllocOp> {
  EliminateAllocInIfWithCopyPattern(MLIRContext *ctx, DominanceInfo &domInfo)
      : OpRewritePattern(ctx), domInfo(domInfo) {}

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Step 1 — Alloc must live in an `scf.if` then-block that has an else.
    Block *thenBlock = allocOp->getBlock();
    auto ifOp = dyn_cast_or_null<scf::IfOp>(thenBlock->getParentOp());
    if (!ifOp || !ifOp.elseBlock() || ifOp.thenBlock() != thenBlock)
      return failure();

    // Step 2 — Find the unique yield-operand position forwarding this
    // alloc.  Bail when yielded at multiple positions; we don't model
    // distinct destinations per position.
    auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
    std::optional<unsigned> resultIdx;
    for (auto [i, operand] : llvm::enumerate(thenYield.getOperands())) {
      if (operand != allocOp.getResult())
        continue;
      if (resultIdx)
        return failure();
      resultIdx = i;
    }
    if (!resultIdx)
      return failure();

    // Step 3 — The if-result must be consumed by exactly one `memref.copy`
    // whose target the else-branch yields directly (so the copy is a no-op
    // on the else path).
    Value ifResult = ifOp.getResult(*resultIdx);
    if (!ifResult.hasOneUse())
      return failure();
    auto copyOp = dyn_cast<memref::CopyOp>(*ifResult.user_begin());
    if (!copyOp || copyOp.getSource() != ifResult)
      return failure();
    Value dest = copyOp.getTarget();
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (elseYield.getOperand(*resultIdx) != dest)
      return failure();

    // Step 4 — Alloc must have a writer besides the yield, and every
    // non-yield use must be a write-only chain root.
    if (llvm::none_of(allocOp->getUsers(), [&](Operation *u) {
          return u != thenYield.getOperation();
        }))
      return failure();
    if (!isWriteOnlyUseChain(allocOp.getResult(), thenYield.getOperation()))
      return failure();

    // Step 5 — `dest` must dominate the alloc (cast inserted at allocOp
    // must be well-formed).
    if (!domInfo.properlyDominates(dest, allocOp.getOperation()))
      return failure();

    auto allocType = allocOp.getType();
    auto destType = cast<MemRefType>(dest.getType());
    // In-place replacement cannot propagate `dest`'s offset into existing
    // view result types.
    if (offsetWouldBeDropped(allocType, destType))
      return failure();
    Value replacement =
        buildDestReplacement(rewriter, dest, allocType, allocOp);
    if (!replacement)
      return failure();

    rewriter.replaceAllUsesWith(allocOp.getResult(), replacement);
    rewriter.eraseOp(allocOp);
    rewriter.eraseOp(copyOp);
    DBG("EliminateAllocInIfWithCopy: redirected scf.if then-branch writes "
        "to destination");
    return success();
  }

private:
  DominanceInfo &domInfo;
};

} // end anonymous namespace

namespace mlir::hexagon {

void populateCopyCanonicalizationAllocPatterns(RewritePatternSet &patterns,
                                               DominanceInfo &domInfo) {
  patterns.add<RedirectAllocWritesToDestinationPattern,
               EliminateAllocInIfWithCopyPattern>(patterns.getContext(),
                                                  domInfo);
}

} // namespace mlir::hexagon
