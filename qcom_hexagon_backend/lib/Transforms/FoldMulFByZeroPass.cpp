//===- FoldMulFByZeroPass.cpp - Fold x * 0.0 with IEEE correctness --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Replaces `x * 0.0` (compile-time constant zero on RHS) with an
// IEEE-754-correct expansion that handles three cases:
//
//   1. x is NaN   -> result = x          (preserve NaN payload)
//   2. x is ±inf  -> result = qNaN       (inf * 0.0 is an invalid operation)
//   3. otherwise   -> result = ±0.0      (zero with sign of x)
//
// AddFastMathPass tags arith.mulf with `reassoc`, which lets the
// canonicalizer fold x * 0.0 -> 0.0 -- losing NaN propagation, signed
// zero, and the inf * 0.0 = NaN requirement. This pass preempts that
// fold with a correct branchless expansion.
//
//   %is_nan  = arith.cmpf uno, %x, %x        // true iff x is NaN
//   %abs_x   = math.absf %x                  // |x|
//   %inf     = arith.constant +inf
//   %is_inf  = arith.cmpf oeq, %abs_x, %inf  // true iff x is ±inf
//   %nan     = arith.constant qNaN             // quiet NaN (type-generic)
//   %szero   = math.copysign %zero, %x       // signed zero matching sign of x
//   %sel_inf = arith.select %is_inf, %nan, %szero  // inf -> NaN, else ±0
//   %result  = arith.select %is_nan, %x, %sel_inf  // NaN -> x (preserves
//                                                   //   payload), else sel_inf
//
// No fastmath flags are required.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"

namespace {

struct FoldMulFByZero : public mlir::OpRewritePattern<mlir::arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::MulFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only fire when RHS is a compile-time zero constant.
    if (!mlir::matchPattern(op.getRhs(), mlir::m_AnyZeroFloat()))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::Value x = op.getLhs();
    mlir::Value zero = op.getRhs(); // the constant 0.0

    // NaN * 0.0 must propagate NaN (IEEE 754).
    mlir::Value isNaN = mlir::arith::CmpFOp::create(
        rewriter, loc, mlir::arith::CmpFPredicate::UNO, x, x);

    // ±inf * 0.0 must produce NaN (IEEE 754 invalid operation).
    mlir::Type resTy = op.getType();
    auto floatTy =
        mlir::cast<mlir::FloatType>(mlir::getElementTypeOrSelf(resTy));
    const auto &sem = floatTy.getFloatSemantics();

    mlir::Value absX = mlir::math::AbsFOp::create(rewriter, loc, x);
    mlir::Value inf = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::FloatAttr::get(floatTy, llvm::APFloat::getInf(sem)));
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(resTy))
      inf = mlir::vector::BroadcastOp::create(rewriter, loc, vecTy, inf);
    mlir::Value isInf = mlir::arith::CmpFOp::create(
        rewriter, loc, mlir::arith::CmpFPredicate::OEQ, absX, inf);

    mlir::Value nan = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::FloatAttr::get(floatTy, llvm::APFloat::getNaN(sem)));
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(resTy))
      nan = mlir::vector::BroadcastOp::create(rewriter, loc, vecTy, nan);

    // -x * 0.0 must yield -0.0 (IEEE 754 signed-zero matching sign of x).
    mlir::Value signedZero =
        mlir::math::CopySignOp::create(rewriter, loc, zero, x);

    // select(isInf, NaN, signedZero) handles the infinity case;
    // select(isNaN, x, ...) handles NaN propagation.
    mlir::Value selInf =
        mlir::arith::SelectOp::create(rewriter, loc, isInf, nan, signedZero);
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, isNaN, x, selInf);
    return mlir::success();
  }
};

struct FoldMulFByZeroPass
    : public mlir::PassWrapper<FoldMulFByZeroPass, mlir::OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldMulFByZeroPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::math::MathDialect,
                    mlir::vector::VectorDialect>();
  }

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
    return "Fold x * 0.0 (constant zero RHS) into an IEEE-correct "
           "NaN-preserving, signed-zero-preserving expansion.";
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
