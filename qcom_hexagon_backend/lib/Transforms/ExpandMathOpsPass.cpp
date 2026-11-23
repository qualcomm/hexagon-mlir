//===- ExpandMathOpsPass.cpp - expand math ops ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Use available math expand patterns.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/SignInference.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "expand-math-ops"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_EXPANDMATHOPS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// IEEE 754: isNaN(x) iff (bitcast(x) & ~signBit) > infBits.
// Returns an invalid Value if the element type is unsupported.
static Value buildIsNan(PatternRewriter &rewriter, Location loc, Value x) {
  auto vecTy = cast<VectorType>(x.getType());
  Type elemTy = vecTy.getElementType();

  unsigned width;
  uint64_t mask, inf;
  if (elemTy.isF32())
    width = 32, mask = 0x7FFFFFFFu, inf = 0x7F800000u;
  else if (elemTy.isF16())
    width = 16, mask = 0x7FFFu, inf = 0x7C00u;
  else
    return {};

  MLIRContext *ctx = rewriter.getContext();
  Type iElem = IntegerType::get(ctx, width);
  Type iVec = VectorType::get(vecTy.getShape(), iElem);

  auto splat = [&](uint64_t val) -> Value {
    return arith::ConstantOp::create(
        rewriter, loc, iVec,
        SplatElementsAttr::get(cast<VectorType>(iVec),
                               rewriter.getIntegerAttr(iElem, val)));
  };

  Value bits = arith::BitcastOp::create(rewriter, loc, iVec, x);
  Value masked = arith::AndIOp::create(rewriter, loc, bits, splat(mask));
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, masked,
                               splat(inf));
}

// une(x, x) — the isNaN idiom. Both operands are the same SSA value, so
// one(x, x) is always false and the result reduces to pure isNaN(x).
struct ExpandVecIsNanPattern : public OpRewritePattern<arith::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPredicate() != arith::CmpFPredicate::UNE)
      return failure();
    if (!isa<VectorType>(op.getLhs().getType()))
      return failure();
    if (op.getLhs() != op.getRhs())
      return failure();

    Value result = buildIsNan(rewriter, op.getLoc(), op.getLhs());
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

// une(a, b) — general case on vectors. HVX vectors only support ordered float
// compares, so we decompose: une(a,b) = isNaN(a) | isNaN(b) | one(a,b).
struct ExpandVecUnePattern : public OpRewritePattern<arith::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPredicate() != arith::CmpFPredicate::UNE)
      return failure();
    if (!isa<VectorType>(op.getLhs().getType()))
      return failure();
    if (op.getLhs() == op.getRhs())
      return failure(); // handled by ExpandVecIsNanPattern

    Location loc = op.getLoc();
    Value isNanA = buildIsNan(rewriter, loc, op.getLhs());
    Value isNanB = buildIsNan(rewriter, loc, op.getRhs());
    if (!isNanA || !isNanB)
      return failure();

    Value one = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::ONE,
                                      op.getLhs(), op.getRhs());
    Value result = arith::OrIOp::create(rewriter, loc, isNanA, isNanB);
    result = arith::OrIOp::create(rewriter, loc, result, one);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Expand math.powf(base, exp) to math.exp(arith.mulf(exp, math.log(base)))
// when the base is provably positive.  The expansion is only valid for base >
// 0: log(base) is NaN for base <= 0, so unknown-sign bases are left intact and
// fall through to UseHexagonRoutines (_hexagon_runtime_pow__vf).
struct ExpandPowFIfPositiveBase : public OpRewritePattern<math::PowFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::PowFOp op,
                                PatternRewriter &rewriter) const override {
    if (inferSign(op.getLhs()) != Sign::Positive)
      return failure();

    Location loc = op.getLoc();
    Value logBase = math::LogOp::create(rewriter, loc, op.getLhs());
    Value scaled = arith::MulFOp::create(rewriter, loc, op.getRhs(), logBase);
    rewriter.replaceOpWithNewOp<math::ExpOp>(op, scaled);
    return success();
  }
};

struct ExpandMathOpsPass : public ::impl::ExpandMathOpsBase<ExpandMathOpsPass> {
  using ExpandMathOpsBase::ExpandMathOpsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    // Expand all math ops except powf: powf expands to exp(y*log(x)) which
    // produces NaN for negative bases. We leave math.powf intact so that
    // ConvertMathToLLVMPass lowers it to llvm.intr.pow (LLVM::PowOp), which
    // UseHexagonRoutines then routes to _hexagon_runtime_pow__vf — a two-arg
    // vector routine that handles negative bases correctly. For provably
    // positive bases, ExpandPowFIfPositiveBase below handles the rewrite.
    //
    // Note: populateExpansionPatterns' filter strips the "math." dialect prefix
    // before matching, so we pass the bare mnemonic strings here.
    math::populateExpansionPatterns(patterns, {
                                                  StringRef("ctlz"),
                                                  StringRef("sinh"),
                                                  StringRef("cosh"),
                                                  StringRef("tan"),
                                                  StringRef("tanh"),
                                                  StringRef("asinh"),
                                                  StringRef("acosh"),
                                                  StringRef("atanh"),
                                                  StringRef("fma"),
                                                  StringRef("ceil"),
                                                  StringRef("exp2"),
                                                  StringRef("round"),
                                                  StringRef("roundeven"),
                                                  StringRef("rsqrt"),
                                                  StringRef("clampf"),
                                              });
    if (expandUneOnVectors)
      patterns.add<ExpandVecIsNanPattern, ExpandVecUnePattern>(
          funcOp.getContext());
    patterns.add<ExpandPowFIfPositiveBase>(funcOp.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createExpandMathOpsPass() {
  return std::make_unique<ExpandMathOpsPass>();
}
