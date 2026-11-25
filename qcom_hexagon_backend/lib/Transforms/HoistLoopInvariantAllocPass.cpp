//===- HoistLoopInvariantAllocPass.cpp ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace hexagon;

#define DEBUG_TYPE "hexagon-hoist-loop-invariant-alloc"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_HEXAGONHOISTLOOPINVARIANTALLOC
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Hoist a static-shape memref.alloc or memref.alloca out of an enclosing
/// scf.for when the allocation does not escape its parent block. The
/// allocation may sit directly in the for-body or inside an scf.if nested in
/// the for.
template <typename AllocOp>
struct HoistLoopInvariantAllocPattern final : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto memrefType = cast<MemRefType>(allocOp->getResult(0).getType());
    if (!memrefType.hasStaticShape())
      return failure();

    Operation *parentOp = allocOp->getParentOp();
    scf::ForOp forOp;

    if (auto directForOp = dyn_cast<scf::ForOp>(parentOp)) {
      forOp = directForOp;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      if (auto enclosingFor = dyn_cast<scf::ForOp>(ifOp->getParentOp()))
        forOp = enclosingFor;
    }

    if (!forOp)
      return failure();

    // Determine whether the buffer (or any view-like alias of it) escapes the
    // loop iteration.
    auto funcOp = allocOp->template getParentOfType<FunctionOpInterface>();
    if (!funcOp)
      return failure();
    BufferViewFlowAnalysis analysis(funcOp);

    Value allocResult = allocOp->getResult(0);
    for (Value alias : analysis.resolve(allocResult)) {
      for (Operation *user : alias.getUsers()) {
        // A dealloc is an ownership boundary, not an ordinary use. If it
        // remains inside the loop, hoisting the allocation would create one
        // allocation with one free per iteration.
        if (isa<memref::DeallocOp>(user))
          return failure();

        // Used outside the loop => the buffer outlives one iteration, so
        // a single hoisted allocation would change semantics.
        if (!forOp->isAncestor(user))
          return failure();
        // Yielded out of an enclosed region (an inner scf.if merge, or a
        // loop-carried scf.for yield) => callers depend on a fresh
        // per-iteration buffer.
        if (auto yieldOp = dyn_cast<scf::YieldOp>(user))
          if (llvm::is_contained(yieldOp.getOperands(), alias))
            return failure();
      }
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);
    Operation *hoistedAlloc = rewriter.clone(*allocOp);
    rewriter.replaceAllUsesWith(allocOp->getResult(0),
                                hoistedAlloc->getResult(0));
    rewriter.eraseOp(allocOp);
    DBG("Hoisted loop-invariant alloc before scf.for: " << *hoistedAlloc);
    return success();
  }
};

/// Hoist the suffix zero-fill of a staging alloc out of a loop when the
/// prefix is completely overwritten by a memref.copy on every iteration.
///
/// The key insight: after `linalg.fill ins(%cst) outs(%alloc)` followed by
/// `memref.copy %src, %alloc[0:M]`, the buffer state is:
///   alloc[0:M]   = values from the copy  (changes every iteration)
///   alloc[M:N]   = zeros from the fill   (NEVER changes — always zero)
///
/// Therefore the fill of alloc[M:N] only needs to happen once, before the
/// outermost enclosing loop.  The per-iteration fill of alloc[0:M] is
/// redundant because the copy overwrites it completely.
///
/// Example:
///   ```
///   %alloc = memref.alloc() : memref<128xf32>   // hoisted
///   scf.for %interval = ... {
///     scf.for %pt = ... {
///       linalg.fill ins(%zero) outs(%alloc)
///       %sv_src = memref.subview %feat_ptr[0][80][1]
///       %sv_dst = memref.subview %alloc[0][80][1]
///       memref.copy %sv_src, %sv_dst
///     }
///   }
///   ```
/// becomes:
///   ```
///   %alloc = memref.alloc() : memref<128xf32>
///   %sv_suffix = memref.subview %alloc[80][48][1]
///   linalg.fill ins(%zero) outs(%sv_suffix)      // zero 48 floats once
///   scf.for %interval = ... {
///     scf.for %pt = ... {
///       %sv_src = memref.subview %feat_ptr[0][80][1]
///       %sv_dst = memref.subview %alloc[0][80][1]
///       memref.copy %sv_src, %sv_dst
///     }
///   }
///   ```
struct HoistPartialFillPattern final : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Returns the outermost scf.for ancestor of `op`, or null if none.
  static scf::ForOp getOutermostEnclosingFor(Operation *op) {
    scf::ForOp result;
    Operation *parent = op->getParentOp();
    while (parent) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        result = forOp;
      parent = parent->getParentOp();
    }
    return result;
  }

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (fillOp.getOutputs().size() != 1)
      return failure();
    auto allocOp = fillOp.getOutputs()[0].getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    auto allocType = cast<MemRefType>(allocOp->getResult(0).getType());
    if (!allocType.hasStaticShape())
      return failure();
    if (allocType.getRank() != 1)
      return failure();

    int64_t N = allocType.getShape()[0]; // nosemgrep: reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-reject-bounds-check,reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-allow-bounds-check

    Block *fillBlock = fillOp->getBlock();
    if (allocOp->getBlock() == fillBlock)
      return failure();

    scf::ForOp outermostFor = getOutermostEnclosingFor(fillOp);
    if (!outermostFor)
      return failure();

    // Find a memref.copy in the same block that writes to a prefix subview
    // of the alloc: subview(%alloc)[0][M][1] with static M < N.
    memref::CopyOp copyOp;
    int64_t M = -1;

    for (Operation &op : *fillBlock) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy)
        continue;

      auto sv = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!sv || sv.getSource() != allocOp->getResult(0))
        continue;

      auto offsets = sv.getStaticOffsets();
      auto sizes = sv.getStaticSizes();
      auto strides = sv.getStaticStrides();
      if (offsets.size() != 1 || sizes.size() != 1 || strides.size() != 1)
        continue;
      if (offsets[0] != 0 || strides[0] != 1)
        continue;
      if (sizes[0] == ShapedType::kDynamic || sizes[0] >= N)
        continue;

      copyOp = copy;
      M = sizes[0];
      break;
    }

    if (!copyOp || M < 0)
      return failure();

    // Verify that no other op in the loop body writes to the alloc (the
    // suffix that we are about to hoist a one-time fill for).
    auto funcOp = fillOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp)
      return failure();
    BufferViewFlowAnalysis aliases(funcOp);
    auto aliasSet = aliases.resolve(allocOp->getResult(0));

    auto writesAnyAlias = [&](Operation *op) {
      for (Value alias : aliasSet)
        if (hasEffect<MemoryEffects::Write>(op, alias))
          return true;
      return false;
    };

    for (Operation &op : *fillBlock) {
      if (&op == fillOp.getOperation() || &op == copyOp.getOperation())
        continue;
      if (writesAnyAlias(&op))
        return failure();
      bool nestedWrite = false;
      for (Region &region : op.getRegions())
        region.walk([&](Operation *nested) {
          if (writesAnyAlias(nested))
            nestedWrite = true;
        });
      if (nestedWrite)
        return failure();
    }

    // Hoist the existing fill before the outermost loop and retarget it at a
    // one-time zero-fill of the suffix.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(outermostFor);

    auto suffixType = cast<MemRefType>(memref::SubViewOp::inferResultType(
        allocType,
        /*staticOffsets=*/ArrayRef<int64_t>{M},
        /*staticSizes=*/ArrayRef<int64_t>{N - M},
        /*staticStrides=*/ArrayRef<int64_t>{1}));

    auto suffixSubview = memref::SubViewOp::create(
        rewriter, fillOp.getLoc(), suffixType, allocOp->getResult(0),
        /*offsets=*/ArrayRef<OpFoldResult>{rewriter.getIndexAttr(M)},
        /*sizes=*/ArrayRef<OpFoldResult>{rewriter.getIndexAttr(N - M)},
        /*strides=*/ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});

    rewriter.moveOpBefore(fillOp, outermostFor);
    rewriter.modifyOpInPlace(fillOp, [&] {
      fillOp.getOutputsMutable().assign(suffixSubview.getResult());
    });

    DBG("HoistPartialFill: hoisted suffix zero-fill ["
        << M << ":" << N
        << "] before outermost loop, removed per-iteration fill (" << N
        << " floats → " << (N - M) << " floats, once)");
    return success();
  }
};

struct HexagonHoistLoopInvariantAllocPass
    : public ::impl::HexagonHoistLoopInvariantAllocBase<
          HexagonHoistLoopInvariantAllocPass> {
  explicit HexagonHoistLoopInvariantAllocPass(
      ArrayRef<TypeID> allocationTypes) {
    for (TypeID allocationType : allocationTypes)
      allocationTypeIDs.insert(allocationType);
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    if (allocationTypeIDs.count(TypeID::get<memref::AllocOp>())) {
      patterns.add<HoistLoopInvariantAllocPattern<memref::AllocOp>>(
          patterns.getContext());
      patterns.add<HoistPartialFillPattern>(patterns.getContext());
    }
    if (allocationTypeIDs.count(TypeID::get<memref::AllocaOp>()))
      patterns.add<HoistLoopInvariantAllocPattern<memref::AllocaOp>>(
          patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

private:
  llvm::SmallDenseSet<TypeID, 2> allocationTypeIDs;
};

template <typename... AllocOps>
std::unique_ptr<OperationPass<func::FuncOp>> createAllocHoistPass() {
  return std::make_unique<HexagonHoistLoopInvariantAllocPass>(
      ArrayRef<TypeID>{TypeID::get<AllocOps>()...});
}

} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonHoistLoopInvariantAllocPass() {
  return createAllocHoistPass<memref::AllocOp, memref::AllocaOp>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonHoistLoopInvariantAllocaPass() {
  return createAllocHoistPass<memref::AllocaOp>();
}
