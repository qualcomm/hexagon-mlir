//===- FoldMulFByZeroPass.cpp - Pass to optimize x * 0 => 0 ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct FoldMulFByZero : public mlir::OpRewritePattern<mlir::arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::MulFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check fastmath flags (need nnan and nsz)
    if (!mlir::arith::bitEnumContainsAll(op.getFastmath(),
                                         mlir::arith::FastMathFlags::nnan |
                                             mlir::arith::FastMathFlags::nsz)) {
      return mlir::failure();
    }

    // Check RHS: x * 0.0 => 0.0 (assume canonical form has constant on RHS)
    if (mlir::matchPattern(op.getRhs(), mlir::m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct FoldMulFByZeroPass
    : public mlir::PassWrapper<FoldMulFByZeroPass, mlir::OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldMulFByZeroPass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldMulFByZero>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const final { return "fold-mulf-by-zero"; }

  mlir::StringRef getDescription() const final {
    return "Fold multiply-by-zero: x * 0.0 = 0.0 (requires fastmath)";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createFoldMulFByZeroPass() {
  return std::make_unique<FoldMulFByZeroPass>();
}

} // namespace hexagon
} // namespace mlir
