//===----------- SmallExponentToMultiplyPass.cpp --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern to convert math.powi operations with small
// integer exponents into sequences of multiplications.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "small-exponent-to-multiply"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SMALLEXPONENTTOMULTIPLY
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Return 1.0 as a constant of the same (scalar or vector float) type as 'ty'.
static Value buildOne(Location loc, Type ty, PatternRewriter &rewriter) {
  if (auto vt = dyn_cast<VectorType>(ty)) {
    auto ft = dyn_cast<FloatType>(vt.getElementType());
    assert(ft && "expected vector of float type");
    auto oneElem = rewriter.getFloatAttr(ft, 1.0);
    auto denseOne = DenseElementsAttr::get(vt, oneElem);
    return rewriter.create<arith::ConstantOp>(loc, vt, denseOne);
  }
  auto ft = dyn_cast<FloatType>(ty);
  assert(ft && "scalar base type must be float");
  return rewriter.create<arith::ConstantOp>(loc,
                                            rewriter.getFloatAttr(ft, 1.0));
}

/// Return minimal multiply chain for x^k, k >= 0. (k == 0 =>1, k == 1 => x, k
/// == 2 => x*x, etc), using exponentiation by squaring.
static Value buildPowPos(Location loc, Value base, int64_t k,
                         PatternRewriter &rewriter) {
  assert(k >= 0 && "exponent must be non-negative");
  Type ty = base.getType();
  Value one = buildOne(loc, ty, rewriter);
  if (k == 0)
    return one;
  if (k == 1)
    return base;

  Value result = one;
  Value cur = base;

  while (k > 0) {
    if (k & 1) // odd exp
      result = rewriter.create<arith::MulFOp>(loc, result, cur);
    k >>= 1; // divide exp by 2
    if (k)
      cur = rewriter.create<arith::MulFOp>(loc, cur, cur);
  }
  return result;
}

// Try to extract a *small constant integer* exponent from "RHS".
/// Returns the small constant integer exponent if found, failure() otherwise.
/// Supports
/// - IntegerAttr (scalar fpowi)
/// - DenseElementsAttr with splat value (vector fpowi)
static FailureOr<int64_t> getSmallConstExponent(Value rhs, int64_t cutoff) {
  if (auto c = rhs.getDefiningOp<arith::ConstantOp>()) {
    Attribute vAttr = c.getValue(); // could be scalar or vector
    if (auto intAttr = dyn_cast<IntegerAttr>(vAttr)) {
      int64_t n = intAttr.getInt();
      if (std::llabs(n) <= cutoff)
        return n;
      return failure();
    }
    if (auto dense = dyn_cast<DenseElementsAttr>(vAttr)) {
      if (!dense.isSplat())
        return failure(); // require splat for vectors
      // Handle index/integer element types.
      if (isa<IntegerType>(dense.getElementType()) ||
          isa<IndexType>(dense.getElementType())) {
        int64_t n = dense.getSplatValue<APInt>().getSExtValue();
        if (std::llabs(n) <= cutoff)
          return n;
        return failure();
      }
    }
  }
  return failure();
}

struct SmallExponentToMultiplyPattern : public OpRewritePattern<math::FPowIOp> {
  SmallExponentToMultiplyPattern(MLIRContext *context, int64_t cutoff,
                                 bool allowNeg)
      : OpRewritePattern<math::FPowIOp>(context), cutoff(cutoff),
        allowNeg(allowNeg) {}

  static bool isFloatOrVectorOfFloat(Type ty) {
    if (isa<FloatType>(ty))
      return true;
    if (auto vt = dyn_cast<VectorType>(ty)) {
      return isa<FloatType>(vt.getElementType());
    }
    return false;
  }

  static bool isIntegerOrVectorOfIntegerOrIndex(Type ty) {
    if (isa<IntegerType>(ty) || isa<IndexType>(ty))
      return true;
    if (auto vt = dyn_cast<VectorType>(ty)) {
      Type elemTy = vt.getElementType();
      return isa<IntegerType>(elemTy) || isa<IndexType>(elemTy);
    }
    return false;
  }

  LogicalResult matchAndRewrite(math::FPowIOp op,
                                PatternRewriter &rewriter) const override {
    // Base must be float (scalar or vector).
    Type resTy = op.getType();
    if (!isFloatOrVectorOfFloat(resTy))
      return failure(); // not float base type

    // Exponent must be integer/index (scalar or vector).
    Type rhsTy = op.getRhs().getType();
    if (!isIntegerOrVectorOfIntegerOrIndex(rhsTy))
      return failure(); // not integer exponent type

    FailureOr<int64_t> nOr = getSmallConstExponent(op.getRhs(), cutoff);
    if (failed(nOr))
      return failure();
    int64_t n = *nOr;

    // If negative not allowed, bail out.
    if (n < 0 && !allowNeg)
      return failure();

    Location loc = op.getLoc();
    Value base = op.getLhs();

    // Build multiply chain for abs(n).
    Value mag = buildPowPos(loc, base, std::llabs(n), rewriter);
    if (n >= 0) {
      rewriter.replaceOp(op, mag);
      return success();
    }

    // Negative exponent: build 1.0 / mag.
    Value one = buildOne(loc, resTy, rewriter);
    Value inv = rewriter.create<arith::DivFOp>(loc, one, mag);
    rewriter.replaceOp(op, inv);
    return success();
  }
  int64_t cutoff;
  bool allowNeg;
};

/// Pass to convert math.pow with small integer exponents to multiplications.
struct SmallExponentToMultiplyPass
    : public ::impl::SmallExponentToMultiplyBase<SmallExponentToMultiplyPass> {
  explicit SmallExponentToMultiplyPass(
      const SmallExponentToMultiplyOptions &options)
      : SmallExponentToMultiplyBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect, arith::ArithDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    int64_t exponentCutoff = maxExponent;
    bool allowNegativeExponents = allowNegative;
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SmallExponentToMultiplyPattern>(ctx, exponentCutoff,
                                                 allowNegativeExponents);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createSmallExponentToMultiplyPass(
    const SmallExponentToMultiplyOptions &options) {
  return std::make_unique<SmallExponentToMultiplyPass>(options);
}
