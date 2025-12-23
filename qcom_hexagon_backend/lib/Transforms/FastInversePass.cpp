//===--------- FastInversePass.cpp -- Fast inverse pass  ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern to convert element-wise floating point
// division operations to multiplication by reciprocal using the Quake III Arena
// fast inverse square root algorithm.
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
    return rewriter.create<arith::ConstantOp>(loc, vt, attr);
  }
  // Scalar f32.
  return rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(val));
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
    return rewriter.create<arith::ConstantOp>(loc, i32Vec, attr);
  }
  // Scalar i32.
  return rewriter.create<arith::ConstantIntOp>(loc, val, 32);
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
  Value i = rewriter.create<arith::BitcastOp>(loc, iTy, x);

  // i = magic - (i >> 1)
  Value one = makeI32One(rewriter, loc, fTy);
  Value shr = rewriter.create<arith::ShRUIOp>(loc, i, one);
  Value diff = rewriter.create<arith::SubIOp>(loc, magic, shr);

  // y = bitcast_f32(i)
  Value y = rewriter.create<arith::BitcastOp>(loc, fTy, diff);

  // y = y * (1.5f - (0.5f * x * y * y))
  Value y2 = rewriter.create<arith::MulFOp>(loc, y, y);
  Value xy2 = rewriter.create<arith::MulFOp>(loc, x, y2);
  Value term = rewriter.create<arith::MulFOp>(loc, c0_5, xy2);
  Value inner = rewriter.create<arith::SubFOp>(loc, c1_5, term);
  Value yref = rewriter.create<arith::MulFOp>(loc, y, inner);

  // Second refinement iteration.
  Value y2b = rewriter.create<arith::MulFOp>(loc, yref, yref);
  Value xy2b = rewriter.create<arith::MulFOp>(loc, x, y2b);
  Value termb = rewriter.create<arith::MulFOp>(loc, c0_5, xy2b);
  Value innerb = rewriter.create<arith::SubFOp>(loc, c1_5, termb);
  Value yrefb = rewriter.create<arith::MulFOp>(loc, yref, innerb);

  return yrefb;
}

// Pattern: arith.divf %p, %q --> %p * rsqrt(%q * %q)

struct DivToQuakeRsqrtPattern : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp divOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = divOp.getLhs();
    Value rhs = divOp.getRhs();
    Location loc = divOp.getLoc();
    Type resultType = divOp.getType();

    // Only handle float or vector of float types.
    if (!isF32OrF32Vector(resultType))
      return failure();

    // Check if RHS is a broadcast operation.
    if (auto broadcastOp = rhs.getDefiningOp<vector::BroadcastOp>()) {
      Value scalarDenom = broadcastOp.getSource();
      Type scalarType = scalarDenom.getType();
      Type vectorType = broadcastOp.getType();

      // Compute fast inverse (scalar).
      // 1. Compute denom * denom.
      Value denom2 =
          rewriter.create<arith::MulFOp>(loc, scalarDenom, scalarDenom);
      // 2. Apply fast inverse square root to get 1/denom.
      Value invScalar = buildQuakeRsqrt(rewriter, loc, denom2);
      // 3. Broadcast the inverse to vector type.
      Value invVec =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, invScalar);
      // 4. Multiply lhs with the broadcasted inverse.
      Value mul = rewriter.create<arith::MulFOp>(loc, lhs, invVec);

      rewriter.replaceOp(divOp, mul);
      return success();
    }

    // Fallback: no broadcast, use original method.
    Value rhs2 = rewriter.create<arith::MulFOp>(loc, rhs, rhs);
    Value invRhs = buildQuakeRsqrt(rewriter, loc, rhs2);
    Value newMul = rewriter.create<arith::MulFOp>(loc, lhs, invRhs);
    rewriter.replaceOp(divOp, newMul);
    return success();
  }
};

struct FastInversePass : public ::impl::FastInverseBase<FastInversePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  StringRef getArgument() const final { return "hexagon-fast-inverse"; }
  StringRef getDescription() const final {
    return "Convert element-wise floating point division to multiplication by "
           "reciprocal using the fast inverse square root algorithm.";
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    patterns.insert<DivToQuakeRsqrtPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createFastInversePass() {
  return std::make_unique<FastInversePass>();
}
