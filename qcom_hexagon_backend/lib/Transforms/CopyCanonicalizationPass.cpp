//===- CopyCanonicalizationPass.cpp - erase redundant copies   ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass canonicalizes memory copy operations by eliminating intermediate
// allocations that serve as passthroughs between source and destination(s).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace hexagon;

#define DEBUG_TYPE "copy-canonicalization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_COPYCANONICALIZATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Pattern to eliminate a single intermediate buffer allocation in copy chains.
///
/// This pattern matches on a copy operation that writes to an
/// intermediate allocation, then replaces all subsequent reading copies
/// to directly use the original source, eliminating the intermediate buffer.
///
/// Example transformation:
///  ```
///   %src = ...
///   %intermediate = memref.alloc() : memref<NxT>
///   memref.copy %src, %intermediate          // Writing copy (matched here)
///   memref.copy %intermediate, %dst1         // Reading copy 1
///   memref.copy %intermediate, %dst2         // Reading copy 2
///  ```
/// Becomes:
///  ```
///   %src = ...
///   memref.copy %src, %dst1
///   memref.copy %src, %dst2
///  ```
/// The pattern handles both single and multiple reader cases.
struct IntermediateBufferEliminationPattern final
    : public OpRewritePattern<memref::CopyOp> {
  IntermediateBufferEliminationPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Value target = op.getTarget();
    auto intermediateAlloc = target.getDefiningOp<memref::AllocOp>();
    if (!intermediateAlloc)
      return failure();

    // Validates that all users except this op (writing copy) are copies that
    // read from the intermediate buffer
    SmallVector<memref::CopyOp> readingCopies;
    for (Operation *user : intermediateAlloc->getUsers()) {
      // Skip the current writing copy
      if (user == op.getOperation())
        continue;

      auto copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp)
        return failure();

      // Must be a reading copy (reads from intermediate buffer)
      if (copyOp.getSource() != intermediateAlloc)
        return failure();

      if (copyOp->getBlock() != op->getBlock() || !op->isBeforeInBlock(copyOp))
        return failure();

      readingCopies.push_back(copyOp);
    }

    if (readingCopies.empty())
      return failure();

    for (auto readingCopy : readingCopies) {
      DBG("Replacing reading copy: " << readingCopy);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(readingCopy);
      auto newCopy = rewriter.replaceOpWithNewOp<memref::CopyOp>(
          readingCopy, source, readingCopy.getTarget());
      DBG("With new copy: " << newCopy);
    }

    DBG("Removed writing copy: " << op);
    DBG("Removed intermediate buffer: " << intermediateAlloc);
    rewriter.eraseOp(op);
    rewriter.eraseOp(intermediateAlloc);
    return success();
  }
};

/// This pattern explicitly adds copies if the yielded value does not match the
/// init_arg. This simplifies subsequent buffer deallocation analyses to dynamic
/// memref aliasing via deallocation helper. In conjuction with other passes and
/// patterns, this enables removal of redundant alloc and copies.
///   Example transformation:
///   ```
///   scf.yield %A : memref<10xf32>
///   ```
///   Becomes:
///   ```
///   memref.copy %A, %loop_arg : memref<10xf32> to memref<10xf32>
///   scf.yield %loop_arg : memref<10xf32>
///   ```
struct AddCopyBeforeYieldPattern final : public OpRewritePattern<scf::YieldOp> {
  AddCopyBeforeYieldPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::YieldOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    if (!forOp)
      return failure();

    bool changed = false;
    SmallVector<Value> newYieldOperands =
        llvm::to_vector<4>(yieldOp.getOperands());

    for (const auto &operand : llvm::enumerate(yieldOp.getOperands())) {
      size_t index = operand.index();
      Value yieldedValue = operand.value();

      if (!mlir::isa<MemRefType>(yieldedValue.getType()))
        continue;

      BlockArgument iterArg = forOp.getRegionIterArgs()[index];
      if (yieldedValue == iterArg)
        continue;

      // If the yieldedValue is a memref and is NOT the iter_arg.
      // Then, we need to insert a copy to canonicalize the IR.
      rewriter.setInsertionPoint(yieldOp); // Insert copy before the yieldOp
      auto newCopy = rewriter.create<memref::CopyOp>(yieldOp.getLoc(),
                                                     yieldedValue, iterArg);
      DBG("Added memref.copy before scf.yield: " << newCopy);

      // The new operand for the yield will be the iter_arg itself
      newYieldOperands[index] = iterArg;
      changed = true;
    }

    if (changed) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOperands);
      return success();
    }

    return failure();
  }
};

struct CopyCanonicalizationPass
    : public ::impl::CopyCanonicalizationBase<CopyCanonicalizationPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<IntermediateBufferEliminationPattern>(patterns.getContext());
    patterns.add<AddCopyBeforeYieldPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createCopyCanonicalizationPass() {
  return std::make_unique<CopyCanonicalizationPass>();
}
