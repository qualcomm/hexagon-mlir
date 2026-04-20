//===--------- FastInversePass.cpp -- Fast inverse pass  ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrite patterns to convert element-wise floating point
// division operations to multiplication by a reciprocal using the Quake III
// Arena fast inverse square root algorithm.
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
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fast-inverse"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_FASTINVERSE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Returns true if 'ty' is f32 or vector of f32.
static bool isF32OrF32Vector(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.isF32();
  if (auto vt = dyn_cast<VectorType>(ty))
    if (auto eft = dyn_cast<FloatType>(vt.getElementType()))
      return eft.isF32();
  return false;
}

// Make a float constant (scalar or splat vector) with the
// same "shape" as 'like'.
static Value makeF32LikeConst(PatternRewriter &rewriter, Location loc,
                              Type like, float val) {
  if (auto vt = dyn_cast<VectorType>(like)) {
    auto elemTy = rewriter.getF32Type();
    auto attr = DenseElementsAttr::get(vt, rewriter.getFloatAttr(elemTy, val));
    return arith::ConstantOp::create(rewriter, loc, vt, attr);
  }
  // Scalar f32.
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getF32FloatAttr(val));
}

// Make an i32 constant (scalar or splat vector) with the
// same "shape" as 'like'. (where 'like' is f32 or vector of f32).
static Value makeI32LikeConst(PatternRewriter &rewriter, Location loc,
                              Type like, int32_t val) {
  auto i32Ty = rewriter.getI32Type();
  if (auto vt = dyn_cast<VectorType>(like)) {
    auto i32Vec = VectorType::get(vt.getShape(), i32Ty);
    auto attr =
        DenseElementsAttr::get(i32Vec, rewriter.getIntegerAttr(i32Ty, val));
    return arith::ConstantOp::create(rewriter, loc, i32Vec, attr);
  }
  // Scalar i32.
  return arith::ConstantIntOp::create(rewriter, loc, val, 32);
}

// Make a splat i32 "1" with the same shape as the i32 type
// dervied from 'like' (where 'like' is f32 or vector of f32).
static Value makeI32One(PatternRewriter &rewriter, Location loc, Type like) {
  return makeI32LikeConst(rewriter, loc, like, 1);
}

static Type getI32LikeType(Type likeF32, MLIRContext *ctx) {
  auto i32Ty = IntegerType::get(ctx, 32);
  if (auto vt = dyn_cast<VectorType>(likeF32)) {
    return VectorType::get(vt.getShape(), i32Ty);
  }
  return i32Ty;
}

// Build IR that computes 1/sqrt(x)
// using Quake III's Fast Inverse Square Root algorithm
static Value buildQuakeRsqrt(PatternRewriter &rewriter, Location loc, Value x) {
  // Quake III Arena fast inverse square root implementation.
  // See https://en.wikipedia.org/wiki/Fast_inverse_square_root
  Type fTy = x.getType();
  assert(isF32OrF32Vector(fTy) && "expected f32 or vector of f32");

  // Constants.
  Value magic = makeI32LikeConst(rewriter, loc, fTy, 0x5f3759df);
  Value c1_5 = makeF32LikeConst(rewriter, loc, fTy, 1.5f);
  Value c0_5 = makeF32LikeConst(rewriter, loc, fTy, 0.5f);

  // Bitcast to int.
  Type iTy = getI32LikeType(fTy, rewriter.getContext());
  Value i = arith::BitcastOp::create(rewriter, loc, iTy, x);

  // i = magic - (i >> 1)
  Value one = makeI32One(rewriter, loc, fTy);
  Value shr = arith::ShRUIOp::create(rewriter, loc, i, one);
  Value diff = arith::SubIOp::create(rewriter, loc, magic, shr);

  // y = bitcast_f32(i)
  Value y = arith::BitcastOp::create(rewriter, loc, fTy, diff);
  // At this point, y is the first approximation of 1/sqrt(x)

  // First Newton's refinement iteration (to increase accuracy)
  // y = y * (1.5f - (0.5f * x * y * y))
  Value y2 = arith::MulFOp::create(rewriter, loc, y, y);
  Value xy2 = arith::MulFOp::create(rewriter, loc, x, y2);
  Value term = arith::MulFOp::create(rewriter, loc, c0_5, xy2);
  Value inner = arith::SubFOp::create(rewriter, loc, c1_5, term);
  Value yref = arith::MulFOp::create(rewriter, loc, y, inner);

  // Second Newton's refinement iteration (to increase accuracy)
  Value y2b = arith::MulFOp::create(rewriter, loc, yref, yref);
  Value xy2b = arith::MulFOp::create(rewriter, loc, x, y2b);
  Value termb = arith::MulFOp::create(rewriter, loc, c0_5, xy2b);
  Value innerb = arith::SubFOp::create(rewriter, loc, c1_5, termb);
  Value yrefb = arith::MulFOp::create(rewriter, loc, yref, innerb);

  return yrefb;
}

// Return true if 'value' is the LLVM intrinsic llvm.intr.sqrt
static bool isLLVMSqrt(Value value) {
  if (auto *op = value.getDefiningOp()) {
    return op->getName().getStringRef() == "llvm.intr.sqrt";
  }
  return false;
}

// Returns true if 'value' is produced by a sqrt, where sqrt
// can be the one from the math dialect, or the llvm intrinsics
static bool isSqrt(Value value) {
  DBG("Checking if this value is a sqrt... \n");

  if (value.getDefiningOp<math::SqrtOp>()) {
    DBG("  -> math.sqrt found! \n");
    return true;
  }

  if (isLLVMSqrt(value)) {
    DBG("  -> llvm.intr.sqrt found! \n");
    return true;
  }

  DBG("  -> No, this value isn't a sqrt. \n");
  return false;
}

// Pattern: arith.divf %p, %q --> %p * Quake_rsqrt(%q * %q)
// that avoids a division (expensive and without HVX support),
// by instead using one multiplication and the inexpensive Quake_rsqrt().
// Note that we do not apply this pattern when %q is of the form sqrt(%a),
// to avoid the anti-optimization of getting back to %a, to then circle
// back to something related to sqrt(%a) which we already had to start with
struct DivToQuakeAndSquaringAndMult : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp divOp,
                                PatternRewriter &rewriter) const override {
    Value num = divOp.getLhs();
    Value denom = divOp.getRhs();
    Location loc = divOp.getLoc();
    Type resultType = divOp.getType();

    // Only handle float or vector of float types.
    if (!isF32OrF32Vector(resultType))
      return failure();

    // VECTORIZED CASE: LHS is "x / broadcast(scalarDenom)"
    // with scalarDenom NOT of the form sqrt(a), and the rewrite will be:
    // num / broadcast(scalarDenom) -->
    //       num * broadcast(Quake_rsqrt(scalarDenom*scalarDenom))
    // -----------------------------------------------------------
    if (auto broadcastOp = denom.getDefiningOp<vector::BroadcastOp>()) {
      Value scalarDenom = broadcastOp.getSource();

      // Do NOT rewrite x / broadcast(sqrt(a)), which would otherwise lead to
      // an anti-optimization: first getting back to 'a' by squaring it, to then
      // circle back to something related to sqrt(a) which we already had
      if (isSqrt(scalarDenom))
        return failure();

      Type vectorType = broadcastOp.getType();

      // 1. Compute denom * denom.
      Value denom2 =
          arith::MulFOp::create(rewriter, loc, scalarDenom, scalarDenom);
      // 2. Apply fast inverse square root to get 1/(denom*denom).
      Value invScalar = buildQuakeRsqrt(rewriter, loc, denom2);
      // 3. Broadcast the inverse to vector type.
      Value invVec =
          vector::BroadcastOp::create(rewriter, loc, vectorType, invScalar);
      // 4. Multiply num with the broadcasted inverse.
      Value mul = arith::MulFOp::create(rewriter, loc, num, invVec);

      rewriter.replaceOp(divOp, mul);
      return success();
    }

    // SCALAR CASE: LHS is "num / denom"
    // with denom NOT of the form sqrt(a), and the rewrite will be:
    // num / denom --> num * Quake_rsqrt(denom*denom)
    // ----------------------------------------
    // Do NOT rewrite x / sqrt(a), which would otherwise lead to
    // an anti-optimization: first getting back to 'a' by squaring it, to then
    // circle back to something related to sqrt(a) which we already had
    if (isSqrt(denom))
      return failure();

    // 1. Compute denom * denom.
    Value denom2 = arith::MulFOp::create(rewriter, loc, denom, denom);
    // 2. Apply fast inverse square root to get 1/(denom*denom).
    Value invDenom2 = buildQuakeRsqrt(rewriter, loc, denom2);
    // 3 Multiply num with the inverse.
    Value newMul = arith::MulFOp::create(rewriter, loc, num, invDenom2);
    rewriter.replaceOp(divOp, newMul);
    return success();
  }
};

// Returns true is the given value 'v' is the fp32 value 1.0f
// or in the case of a dense, a repetition of the value 1.0f everywhere.
// This is a utility function for the second pattern that will follow
static bool isOneF32(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    // Case of a single float value
    if (auto attr = dyn_cast<FloatAttr>(cst.getValue()))
      return attr.getType().isF32() && attr.getValueAsDouble() == 1.0;
    // Case of a dense
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
      if (!dense.getElementType().isF32())
        return false;
      for (auto val : dense.getValues<float>()) // Check all the elements
        if (val != 1.0f)
          return false;
      return true;
    }
  }
  return false;
}

// Extract the operand of a square root operation, whether it's a math.sqrt or
// an llvm.intr.sqrt. Fails if the input 'x' was none of them. Need to call
// isSqrt(x) before calling this function
Value extractValueFromSqrt(Value x) {
  // We will extract 'a' from x = sqrt(a)
  Value a;

  // Case where we extract it from math.sqrt
  if (auto ms = x.getDefiningOp<math::SqrtOp>()) {
    a = ms.getOperand();
    return a;
  }

  // Case where we extract it from llvm.intr.sqrt
  auto *op = x.getDefiningOp();
  assert(op && "No defining op found");
  assert(op->getNumOperands() == 1 &&
         "llvm.intr.sqrt should have exactly one operand");
  a = op->getOperand(0);
  return a;
}

// 2 more specialized rewrite rules that are more efficient that the general one
// a) The "most" specialized one:
// arith.divf one, sqrt(%a) --> Quake_rsqrt(%a)
// b) The "more" specialized one (whose RHS contains an extra mult):
// arith.divf %num, sqrt(%a) --> %num * Quake_rsqrt(%a)
// They share the same 2 benefits:
// 1. They replace an expensive sqrt() op by the inexpensive Quake_rsqrt() op,
// 2. They entirely remove a division (expensive and without HVX support)
// Performance justifications:
// a) Knowing that Cost(Quake_rsqrt) < Cost(sqrt), the rewrite rule a)
// is always an optimization, since it is trivially provable that
// Cost(Quake_rsqrt(%a)) < Cost(sqrt) + Cost(divf)
// b) Additionally knowing that cost(mult) < cost(divf), it is also trivially
// provable that Cost(Quake_rsqrt(%a)) + Cost(mult) < Cost(sqrt) + Cost(divf)
struct DivWithSqrtDenom : public OpRewritePattern<arith::DivFOp> {

  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp divOp,
                                PatternRewriter &rewriter) const override {
    Value num = divOp.getLhs();
    Value denom = divOp.getRhs();
    Location loc = divOp.getLoc();
    Type ty = divOp.getType();

    // Only f32 or vector<f32>
    if (!isF32OrF32Vector(ty))
      return failure();

    // VECTORIZED CASE: LHS is "num / broadcast(sqrt(a))"
    // There will be 2 possibilities:
    // 1 / broadcast(sqrt(a)) --> broadcast(Quake_rsqrt(a))
    // x / broadcast(sqrt(a)) --> x * broadcast(Quake_rsqrt(a))
    // --------------------------------------------------------
    if (auto broadcastOp = denom.getDefiningOp<vector::BroadcastOp>()) {
      Value scalarDenom = broadcastOp.getSource();

      // scalarDenom must be of the form sqrt(a) for this specific pattern
      if (!isSqrt(scalarDenom))
        return failure();

      Type vectorType = broadcastOp.getType();

      Value a = extractValueFromSqrt(scalarDenom);

      // Build Quake_rsqrt(a)
      Value rsqrt = buildQuakeRsqrt(rewriter, loc, a);
      Value broadcastedQuake =
          vector::BroadcastOp::create(rewriter, loc, vectorType, rsqrt);

      // If the numerator is exactly 1.0, we can directly rewrite the whole LHS
      // with the broadcastedQuake
      if (isOneF32(num)) {
        rewriter.replaceOp(divOp, broadcastedQuake);
        return success();
      }

      // Else, when the numerator was anything else, we need to produce a
      // multiplication
      Value mult = arith::MulFOp::create(rewriter, loc, num, broadcastedQuake);
      rewriter.replaceOp(divOp, mult);
      return success();
    }

    // SCALAR CASE: LHS is "num / sqrt(a)"
    // There will be 2 possibilities:
    // 1 / sqrt(a) --> Quake_rsqrt(a)
    // x / sqrt(a) --> x * Quake_rsqrt(a)
    // ----------------------------------
    // denom must be of the form sqrt(a) for this specific pattern
    if (!isSqrt(denom))
      return failure();

    Value a = extractValueFromSqrt(denom);

    // Build Quake_rsqrt(a)
    Value rsqrt = buildQuakeRsqrt(rewriter, loc, a);

    // If the numerator is exactly 1.0, we can directly rewrite the whole LHS
    // with the rsqrt
    if (isOneF32(num)) {
      rewriter.replaceOp(divOp, rsqrt);
      return success();
    }

    // Else, when the numerator was anything else, we need to produce a
    // multiplication
    Value mult = arith::MulFOp::create(rewriter, loc, num, rsqrt);
    rewriter.replaceOp(divOp, mult);
    return success();
  }
};

struct FastInversePass : public ::impl::FastInverseBase<FastInversePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect,
                    math::MathDialect>();
  }
  StringRef getArgument() const final { return "hexagon-fast-inverse"; }
  StringRef getDescription() const final {
    return "Convert element-wise floating point division to multiplication by "
           "reciprocal using the fast inverse square root algorithm, "
           "and related optimizations.";
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    // 1. More specialized rewrite rules (sqrt denominators), producing NO
    // squaring and also NO product when the numerator is the value 1, i.e.:
    // arith.divf one, sqrt(%a) --> Quake_rsqrt(%a)
    // arith.divf x, sqrt(%a) --> x * Quake_rsqrt(%a)
    patterns.insert<DivWithSqrtDenom>(patterns.getContext());
    // -----
    // 2. More general rewrite rule, producing a product and the square of the
    // denom arith.divf %p, %q --> %p * Quake_rsqrt(%q * %q)
    patterns.insert<DivToQuakeAndSquaringAndMult>(patterns.getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createFastInversePass() {
  return std::make_unique<FastInversePass>();
}
