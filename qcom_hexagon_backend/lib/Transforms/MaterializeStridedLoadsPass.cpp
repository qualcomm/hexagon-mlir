//===- MaterializeStridedLoadsPass.cpp ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/CopyCanonicalizationCommon.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace hexagon;

#define DEBUG_TYPE "hexagon-materialize-strided-loads"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_HEXAGONMATERIALIZESTRIDEDLOADS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// HVX vector alignment in bytes (matches kHVXVectorAlignment in
/// MemoryOffsetsPass).  The staging buffer must be HVX-aligned so the
/// unit-stride vector loads/stores it feeds lower to aligned `vmem`.
constexpr int64_t kHVXVectorAlignment = 128;

/// Materialize a 1-D static memref.reinterpret_cast (stride > 1) feeding a
/// vector.transfer_read/write through a subview into a contiguous alloc, so
/// the vector ops use unit-stride loads/stores instead of scalar
/// gathers/scatters.  A read use gets a copy-in `memref.copy %cast, %alloc`
/// (gather); a write use gets a drain `memref.copy %alloc, %cast` (scatter)
/// after the writers; a read-modify-write use gets both.
///
/// Assumes the writers cover the whole materialized block before the drain;
/// a partial-block write would have the drain flush stale buffer slots.
struct StridedMemrefMaterializationPattern final
    : public OpRewritePattern<memref::ReinterpretCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != 1)
      return failure();

    auto stridedLayout = dyn_cast<StridedLayoutAttr>(resultType.getLayout());
    if (!stridedLayout)
      return failure();

    auto strides = stridedLayout.getStrides();
    if (strides.size() != 1)
      return failure();

    int64_t stride = strides[0];
    if (stride <= 1 || stride == ShapedType::kDynamic)
      return failure();

    // Vectorization-prep optimization: only fire when the strided memref
    // actually feeds a vector load/store path; scalar-only users do not
    // benefit.
    bool hasReadUse = false;
    bool hasWriteUse = false;
    for (Operation *user : op.getResult().getUsers()) {
      auto subview = dyn_cast<memref::SubViewOp>(user);
      if (!subview)
        continue;
      for (Operation *svUser : subview.getResult().getUsers()) {
        if (isa<vector::TransferReadOp>(svUser))
          hasReadUse = true;
        else if (isa<vector::TransferWriteOp>(svUser))
          hasWriteUse = true;
      }
    }
    if (!hasReadUse && !hasWriteUse)
      return failure();

    // The drain copy-back is inserted after every writer, so its anchor is the
    // last writer (in the cast's block) in program order.  Compute it up front
    // to bail before mutating if any writer does not come after the cast.
    Operation *drainAnchor = nullptr;
    if (hasWriteUse) {
      Block *castBlock = op->getBlock();
      for (Operation *user : op.getResult().getUsers()) {
        auto subview = dyn_cast<memref::SubViewOp>(user);
        if (!subview)
          continue;
        for (Operation *svUser : subview.getResult().getUsers()) {
          if (!isa<vector::TransferWriteOp>(svUser))
            continue;
          // The anchor must be the writer's ancestor *in the cast's block*, not
          // the writer itself, so the drain is well-formed when writers sit in
          // nested regions (e.g. an scf.for body).
          Operation *ancestor = castBlock->findAncestorOpInBlock(*svUser);
          if (!ancestor || !hexagon::detail::happensBefore(op, ancestor))
            return failure();
          if (!drainAnchor ||
              hexagon::detail::happensBefore(drainAnchor, ancestor))
            drainAnchor = ancestor;
        }
      }
      if (!drainAnchor)
        return failure();
    }

    auto contiguousType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());

    rewriter.setInsertionPointAfter(op);
    auto allocOp = memref::AllocOp::create(
        rewriter, op.getLoc(), contiguousType,
        /*dynamicSizes=*/ValueRange{},
        /*symbolOperands=*/ValueRange{},
        rewriter.getI64IntegerAttr(kHVXVectorAlignment));

    // Skip the copy-in for a pure-write cast: there is nothing to read, and a
    // gather would pull in uninitialized destination memory.
    memref::CopyOp copyInOp;
    if (hasReadUse)
      copyInOp = memref::CopyOp::create(rewriter, op.getLoc(), op.getResult(),
                                        allocOp.getResult());

    memref::CopyOp drainOp;
    if (hasWriteUse) {
      rewriter.setInsertionPointAfter(drainAnchor);
      drainOp = memref::CopyOp::create(rewriter, op.getLoc(),
                                       allocOp.getResult(), op.getResult());
    }

    SmallVector<memref::SubViewOp> subviewsToUpdate;
    for (Operation *user : op.getResult().getUsers()) {
      if (auto sv = dyn_cast<memref::SubViewOp>(user))
        subviewsToUpdate.push_back(sv);
    }

    // Redirect direct (non-subview) uses of the cast to the alloc, but keep
    // the copy-in's source and the drain's target pointing at the cast.
    op.getResult().replaceUsesWithIf(allocOp.getResult(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner != copyInOp && owner != drainOp &&
             !isa<memref::SubViewOp>(owner);
    });

    for (auto subview : subviewsToUpdate) {
      auto newResultType = memref::SubViewOp::inferResultType(
          contiguousType, subview.getStaticOffsets(), subview.getStaticSizes(),
          subview.getStaticStrides());
      rewriter.setInsertionPoint(subview);
      auto newSubview = memref::SubViewOp::create(
          rewriter, subview.getLoc(), cast<MemRefType>(newResultType),
          allocOp.getResult(), subview.getMixedOffsets(),
          subview.getMixedSizes(), subview.getMixedStrides());
      rewriter.replaceOp(subview, newSubview.getResult());
    }

    DBG("Materialized strided reinterpret_cast (stride="
        << stride << ", size=" << resultType.getShape()[0]
        << ", read=" << hasReadUse << ", write=" << hasWriteUse
        << ") into contiguous alloc");
    return success();
  }
};

struct HexagonMaterializeStridedLoadsPass
    : public ::impl::HexagonMaterializeStridedLoadsBase<
          HexagonMaterializeStridedLoadsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StridedMemrefMaterializationPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonMaterializeStridedLoadsPass() {
  return std::make_unique<HexagonMaterializeStridedLoadsPass>();
}
