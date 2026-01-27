//===-- RewriteUBPoisonToZero.cpp -  rewrite ub.poison        -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern to rewrite ub.poison operations to zero
// constants for vector types.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_REWRITEUBPOISONTOZERO
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {
struct UBPoisonPaddingToZeroPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    Value padding = op.getPadding();
    if (!padding)
      return failure();

    if (auto poison = padding.getDefiningOp<ub::PoisonOp>()) {
      Type padTy = padding.getType();

      TypedAttr zeroAttr;
      if (auto intTy = dyn_cast<IntegerType>(padTy))
        zeroAttr = rewriter.getIntegerAttr(intTy, 0);
      else if (auto floatTy = dyn_cast<FloatType>(padTy))
        zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
      else
        return rewriter.notifyMatchFailure(op, "Unsupported padding type");

      // Create a constant zero of the same type as padding
      auto zeroConst =
          rewriter.create<arith::ConstantOp>(op.getLoc(), padTy, zeroAttr);

      // Replace the padding with the zero constant.
      rewriter.modifyOpInPlace(
          op, [&]() { op.getPaddingMutable().assign(zeroConst); });
      return success();
    }
    return failure();
  }
};

struct RewriteUBPoisonToZeroPass
    : public ::impl::RewriteUBPoisonToZeroBase<RewriteUBPoisonToZeroPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ub::UBDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
  }

  StringRef getArgument() const final {
    return "hexagon-rewrite-ub-poison-to-zero";
  }
  StringRef getDescription() const final {
    return "Rewrite ub.poison operations used as padding in "
           "vector.transfer_read to zero constants";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<UBPoisonPaddingToZeroPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createRewriteUBPoisonToZeroPass() {
  return std::make_unique<RewriteUBPoisonToZeroPass>();
}
