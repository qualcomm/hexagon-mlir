//===- LowerCompositePass.cpp - Lower composite dialect ops ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Lowers memref-form composite ops to vectorized scf.for loops targeting
// HVX-width vectors.  Each inner loop processes two vector chunks per
// iteration (step = 2 * VW) to expose instruction-level parallelism on
// the in-order VLIW pipeline.
//
// By recognising high-level ops (softmax, rms_norm, ...) as single
// composite ops we decouple them from the general-purpose lowering
// pipeline and gain direct control over the generated code.  The lowering
// can be iteratively tuned — unrolling, scheduling, memory-access
// strategy, etc. — without affecting or being constrained by the rest of
// the compiler.
//
// Currently implemented:
//   - LowerSoftmaxPattern:   3-loop softmax (max, exp+sum, normalize)
//   - LowerRmsNormPattern:   2-loop RMS norm (square+accumulate, normalize)
//   - LowerLayerNormPattern: 3-loop layer norm (mean, variance, normalize)
//
// Constraints / future work:
//   - Only innermost axis (axis == rank - 1)
//   - Reduction dim must be divisible by vectorWidth * unrollFactor
//   - Remainder handling, multi-axis support, and fp16 are not yet
//     implemented
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/Composite/IR/CompositeDialect.h"
#include "hexagon/Dialect/Composite/IR/CompositeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_LOWERCOMPOSITE
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

using namespace mlir;
using namespace hexagon;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// HVX register width in bytes.
static constexpr int64_t kHVXWidthBytes = 128;

/// Unroll factor for the inner vector loops.
static constexpr int64_t kUnrollFactor = 2;

/// FastMath flags for arithmetic ops.  The composite lowering runs after
/// AddFastMathPass so we must set the flags ourselves.
/// reassoc: enables tree reductions and reassociation in the backend.
/// nnan:    required for min/max to lower to native HVX vmax/vmin instead
///          of scalarized libcalls (see AddFastMathPass.cpp commentary).
static constexpr auto kFMFReassoc = arith::FastMathFlags::reassoc;
static constexpr auto kFMFReassocNnan =
    arith::FastMathFlags::reassoc | arith::FastMathFlags::nnan;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Attach arith fastmath flags to the op that defines `val`.
static void setFastMath(Value val, arith::FastMathFlags fmf) {
  val.getDefiningOp()->setAttr(
      arith::AddFOp::getFastMathAttrName(),
      arith::FastMathFlagsAttr::get(val.getContext(), fmf));
}

/// Holds the shared operands needed by the three softmax loop emitters
/// (max-reduce, exp+sum, normalize) so they don't have to be threaded
/// individually through every call.
struct SoftmaxEmitter {
  PatternRewriter &rewriter;
  Location loc;
  Value input, output;
  VectorType vecTy;
  Value zero, innerUpper, step, vwVal;

  /// Return [outerIVs..., col (+ offset)].
  SmallVector<Value> indicesAt(ValueRange outerIVs, Value col,
                               Value offset) const {
    SmallVector<Value> idx(outerIVs.begin(), outerIVs.end());
    if (offset)
      idx.push_back(arith::AddIOp::create(rewriter, loc, col, offset));
    else
      idx.push_back(col);
    return idx;
  }

  /// Emit the max-reduce loop.  Returns max broadcast to a full vector.
  Value emitMaxReduce(ValueRange outerIVs, Value negInfVec) const {
    auto loop = scf::ForOp::create(rewriter, loc, zero, innerUpper, step,
                                   ValueRange{negInfVec});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value col = loop.getBody()->getArgument(0);
      Value acc = loop.getBody()->getArgument(1);
      auto idx0 = indicesAt(outerIVs, col, nullptr);
      auto idx1 = indicesAt(outerIVs, col, vwVal);

      Value v0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
      Value v1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
      Value localMax =
          arith::MaxNumFOp::create(rewriter, loc, v0, v1, kFMFReassocNnan);
      Value newMax = arith::MaxNumFOp::create(rewriter, loc, acc, localMax,
                                              kFMFReassocNnan);
      scf::YieldOp::create(rewriter, loc, ValueRange{newMax});
    }
    Value maxVec = loop.getResult(0);
    Value maxScalar = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::MAXNUMF, maxVec);
    setFastMath(maxScalar, kFMFReassocNnan);
    return vector::BroadcastOp::create(rewriter, loc, vecTy, maxScalar);
  }

  /// Emit the fused exp + sum-reduce loop.  Stores exp(x-max) into `output`
  /// and returns (1 / sum) broadcast to a full vector.
  Value emitExpSum(ValueRange outerIVs, Value maxBcast, Value zeroVec,
                   Value oneF) const {
    auto loop = scf::ForOp::create(rewriter, loc, zero, innerUpper, step,
                                   ValueRange{zeroVec});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value col = loop.getBody()->getArgument(0);
      Value acc = loop.getBody()->getArgument(1);
      auto idx0 = indicesAt(outerIVs, col, nullptr);
      auto idx1 = indicesAt(outerIVs, col, vwVal);

      Value v0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
      Value shifted0 =
          arith::SubFOp::create(rewriter, loc, v0, maxBcast, kFMFReassoc);
      Value exp0 = math::ExpOp::create(rewriter, loc, shifted0);
      setFastMath(exp0, kFMFReassoc);
      vector::StoreOp::create(rewriter, loc, exp0, output, idx0);

      Value v1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
      Value shifted1 =
          arith::SubFOp::create(rewriter, loc, v1, maxBcast, kFMFReassoc);
      Value exp1 = math::ExpOp::create(rewriter, loc, shifted1);
      setFastMath(exp1, kFMFReassoc);
      vector::StoreOp::create(rewriter, loc, exp1, output, idx1);

      Value sum01 =
          arith::AddFOp::create(rewriter, loc, exp0, exp1, kFMFReassoc);
      Value newSum =
          arith::AddFOp::create(rewriter, loc, acc, sum01, kFMFReassoc);
      scf::YieldOp::create(rewriter, loc, ValueRange{newSum});
    }
    Value sumVec = loop.getResult(0);
    Value sumScalar = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::ADD, sumVec);
    setFastMath(sumScalar, kFMFReassoc);
    Value rcp =
        arith::DivFOp::create(rewriter, loc, oneF, sumScalar, kFMFReassoc);
    return vector::BroadcastOp::create(rewriter, loc, vecTy, rcp);
  }

  /// Emit the normalization loop: output[i] *= (1 / sum).
  void emitNormalize(ValueRange outerIVs, Value rcpBcast) const {
    auto loop = scf::ForOp::create(rewriter, loc, zero, innerUpper, step);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value col = loop.getBody()->getArgument(0);
      auto idx0 = indicesAt(outerIVs, col, nullptr);
      auto idx1 = indicesAt(outerIVs, col, vwVal);

      Value val0 = vector::LoadOp::create(rewriter, loc, vecTy, output, idx0);
      Value norm0 =
          arith::MulFOp::create(rewriter, loc, val0, rcpBcast, kFMFReassoc);
      vector::StoreOp::create(rewriter, loc, norm0, output, idx0);

      Value val1 = vector::LoadOp::create(rewriter, loc, vecTy, output, idx1);
      Value norm1 =
          arith::MulFOp::create(rewriter, loc, val1, rcpBcast, kFMFReassoc);
      vector::StoreOp::create(rewriter, loc, norm1, output, idx1);
    }
  }
};

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

struct LowerSoftmaxPattern : public OpRewritePattern<composite::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(composite::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    // Only lower the memref form (0 results).
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "expected memref form");

    Value input = op.getInput();
    Value output = op.getInit();
    auto memrefTy = cast<MemRefType>(input.getType());
    Type elemTy = memrefTy.getElementType();
    auto floatTy = dyn_cast<FloatType>(elemTy);
    if (!floatTy)
      return rewriter.notifyMatchFailure(op, "expected float element type");

    int64_t rank = memrefTy.getRank();
    if (rank < 1)
      return rewriter.notifyMatchFailure(op, "rank must be >= 1");

    Location loc = op.getLoc();
    int64_t elemBytes = floatTy.getIntOrFloatBitWidth() / 8;
    int64_t vectorWidth = kHVXWidthBytes / elemBytes;
    int64_t unrolledStep = vectorWidth * kUnrollFactor;
    auto vecTy = VectorType::get({vectorWidth}, elemTy);

    // Outer dim upper bounds (all except the reduction axis).
    SmallVector<Value> outerUppers;
    for (int64_t i = 0; i + 1 < rank; ++i) {
      if (memrefTy.isDynamicDim(i))
        outerUppers.push_back(memref::DimOp::create(rewriter, loc, input, i));
      else
        outerUppers.push_back(arith::ConstantIndexOp::create(
            rewriter, loc, memrefTy.getDimSize(i)));
    }

    // Reduction axis upper bound.
    int64_t innerStaticSize = memrefTy.getDimSize(rank - 1);
    Value innerUpper;
    if (ShapedType::isDynamic(innerStaticSize))
      innerUpper = memref::DimOp::create(rewriter, loc, input, rank - 1);
    else
      innerUpper =
          arith::ConstantIndexOp::create(rewriter, loc, innerStaticSize);

    // Index and vector constants.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, unrolledStep);
    Value vwVal = arith::ConstantIndexOp::create(rewriter, loc, vectorWidth);

    auto negInfAP = APFloat::getInf(floatTy.getFloatSemantics(),
                                    /*Negative=*/true);
    Value negInfVec = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(vecTy, negInfAP));
    Value zeroVec = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(vecTy,
                               APFloat::getZero(floatTy.getFloatSemantics())));
    Value oneF = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getFloatAttr(elemTy, 1.0));

    SoftmaxEmitter emitter{rewriter, loc,        input, output, vecTy,
                           zero,     innerUpper, step,  vwVal};

    // Nest scf.for loops over the outer dimensions, then emit the softmax
    // body at the innermost level.
    SmallVector<Value> ivs;
    std::function<void(unsigned)> buildOuter = [&](unsigned d) {
      if (d == outerUppers.size()) {
        Value maxBcast = emitter.emitMaxReduce(ivs, negInfVec);
        Value rcpBcast = emitter.emitExpSum(ivs, maxBcast, zeroVec, oneF);
        emitter.emitNormalize(ivs, rcpBcast);
        return;
      }
      auto loop = scf::ForOp::create(rewriter, loc, zero, outerUppers[d], one);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      buildOuter(d + 1);
      rewriter.setInsertionPointAfter(loop);
      ivs.pop_back();
    };
    buildOuter(0);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerRmsNormPattern
//===----------------------------------------------------------------------===//

struct LowerRmsNormPattern : public OpRewritePattern<composite::RmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(composite::RmsNormOp op,
                                PatternRewriter &rewriter) const override {
    // Only lower the memref form (0 results).
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "expected memref form");

    Value input = op.getInput();
    Value weights = op.getWeights();
    Value output = op.getInit();
    auto memrefTy = cast<MemRefType>(input.getType());
    Type elemTy = memrefTy.getElementType();
    auto floatTy = dyn_cast<FloatType>(elemTy);
    if (!floatTy)
      return rewriter.notifyMatchFailure(op, "expected float element type");

    int64_t rank = memrefTy.getRank();
    if (rank < 1)
      return rewriter.notifyMatchFailure(op, "rank must be >= 1");

    Location loc = op.getLoc();
    int64_t elemBytes = floatTy.getIntOrFloatBitWidth() / 8;
    int64_t vectorWidth = kHVXWidthBytes / elemBytes;
    int64_t unrolledStep = vectorWidth * kUnrollFactor;
    auto vecTy = VectorType::get({vectorWidth}, elemTy);

    // Outer dim upper bounds (all except the reduction axis).
    SmallVector<Value> outerUppers;
    for (int64_t i = 0; i + 1 < rank; ++i) {
      if (memrefTy.isDynamicDim(i))
        outerUppers.push_back(memref::DimOp::create(rewriter, loc, input, i));
      else
        outerUppers.push_back(arith::ConstantIndexOp::create(
            rewriter, loc, memrefTy.getDimSize(i)));
    }

    // Reduction axis upper bound.
    int64_t axisStaticSize = memrefTy.getDimSize(rank - 1);
    Value axisUpper;
    if (ShapedType::isDynamic(axisStaticSize))
      axisUpper = memref::DimOp::create(rewriter, loc, input, rank - 1);
    else
      axisUpper = arith::ConstantIndexOp::create(rewriter, loc, axisStaticSize);

    // N as float for mean computation (from the norm_size attribute, which
    // is the actual element count — may differ from dim size when padded).
    float normSizeValue = op.getNormSize().convertToFloat();
    Value nFloat = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elemTy, static_cast<double>(normSizeValue)));

    float epsValue = op.getEpsilon().convertToFloat();

    // Index and vector constants.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, unrolledStep);
    Value vwVal = arith::ConstantIndexOp::create(rewriter, loc, vectorWidth);

    Value zeroVec = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(vecTy,
                               APFloat::getZero(floatTy.getFloatSemantics())));
    Value oneF = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getFloatAttr(elemTy, 1.0));
    Value epsF = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemTy, epsValue));

    // Helper to compute indices at [outerIVs..., col + offset].
    auto indicesAt = [&](ValueRange outerIVs, Value col,
                         Value offset) -> SmallVector<Value> {
      SmallVector<Value> idx(outerIVs.begin(), outerIVs.end());
      if (offset)
        idx.push_back(arith::AddIOp::create(rewriter, loc, col, offset));
      else
        idx.push_back(col);
      return idx;
    };

    // Emit the RMS norm body for a single row (given outer IVs).
    auto emitRmsNormBody = [&](ValueRange outerIVs) {
      // Loop 1: acc += x[i] * x[i]
      auto sumLoop = scf::ForOp::create(rewriter, loc, zero, axisUpper, step,
                                        ValueRange{zeroVec});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(sumLoop.getBody());
        Value col = sumLoop.getBody()->getArgument(0);
        Value acc = sumLoop.getBody()->getArgument(1);
        auto idx0 = indicesAt(outerIVs, col, nullptr);
        auto idx1 = indicesAt(outerIVs, col, vwVal);

        Value v0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
        Value sq0 = arith::MulFOp::create(rewriter, loc, v0, v0, kFMFReassoc);
        Value v1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
        Value sq1 = arith::MulFOp::create(rewriter, loc, v1, v1, kFMFReassoc);

        Value sum01 =
            arith::AddFOp::create(rewriter, loc, sq0, sq1, kFMFReassoc);
        Value newAcc =
            arith::AddFOp::create(rewriter, loc, acc, sum01, kFMFReassoc);
        scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
      }

      // Horizontal reduce -> mean -> rsqrt scalar.
      Value sumVec = sumLoop.getResult(0);
      Value sumScalar = vector::ReductionOp::create(
          rewriter, loc, vector::CombiningKind::ADD, sumVec);
      setFastMath(sumScalar, kFMFReassoc);

      Value mean =
          arith::DivFOp::create(rewriter, loc, sumScalar, nFloat, kFMFReassoc);
      Value shifted =
          arith::AddFOp::create(rewriter, loc, mean, epsF, kFMFReassoc);
      Value sqrtVal = math::SqrtOp::create(rewriter, loc, shifted);
      setFastMath(sqrtVal, kFMFReassoc);
      Value rsqrt =
          arith::DivFOp::create(rewriter, loc, oneF, sqrtVal, kFMFReassoc);
      Value rsqrtBcast =
          vector::BroadcastOp::create(rewriter, loc, vecTy, rsqrt);

      // Loop 2: out[i] = x[i] * rsqrt * w[i]
      auto normLoop = scf::ForOp::create(rewriter, loc, zero, axisUpper, step);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(normLoop.getBody());
        Value col = normLoop.getBody()->getArgument(0);
        auto idx0 = indicesAt(outerIVs, col, nullptr);
        auto idx1 = indicesAt(outerIVs, col, vwVal);

        // Chunk 0.
        Value x0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
        Value w0 = vector::LoadOp::create(rewriter, loc, vecTy, weights, idx0);
        Value n0 =
            arith::MulFOp::create(rewriter, loc, x0, rsqrtBcast, kFMFReassoc);
        Value out0 = arith::MulFOp::create(rewriter, loc, n0, w0, kFMFReassoc);
        vector::StoreOp::create(rewriter, loc, out0, output, idx0);

        // Chunk 1.
        Value x1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
        Value w1 = vector::LoadOp::create(rewriter, loc, vecTy, weights, idx1);
        Value n1 =
            arith::MulFOp::create(rewriter, loc, x1, rsqrtBcast, kFMFReassoc);
        Value out1 = arith::MulFOp::create(rewriter, loc, n1, w1, kFMFReassoc);
        vector::StoreOp::create(rewriter, loc, out1, output, idx1);
      }
    };

    // Nest scf.for loops over outer dimensions, emit body at innermost level.
    SmallVector<Value> ivs;
    std::function<void(unsigned)> buildOuter = [&](unsigned d) {
      if (d == outerUppers.size()) {
        emitRmsNormBody(ivs);
        return;
      }
      auto loop = scf::ForOp::create(rewriter, loc, zero, outerUppers[d], one);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      buildOuter(d + 1);
      rewriter.setInsertionPointAfter(loop);
      ivs.pop_back();
    };
    buildOuter(0);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerLayerNormPattern
//===----------------------------------------------------------------------===//

struct LowerLayerNormPattern : public OpRewritePattern<composite::LayerNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(composite::LayerNormOp op,
                                PatternRewriter &rewriter) const override {
    // Only lower the memref form (0 results).
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "expected memref form");

    Value input = op.getInput();
    Value weights = op.getWeights();
    Value bias = op.getBias();
    Value output = op.getInit();
    Value meanOut = op.getMeanOut();
    Value rstdOut = op.getRstdOut();
    auto memrefTy = cast<MemRefType>(input.getType());
    Type elemTy = memrefTy.getElementType();
    auto floatTy = dyn_cast<FloatType>(elemTy);
    if (!floatTy)
      return rewriter.notifyMatchFailure(op, "expected float element type");

    int64_t rank = memrefTy.getRank();
    if (rank < 1)
      return rewriter.notifyMatchFailure(op, "rank must be >= 1");

    Location loc = op.getLoc();
    int64_t elemBytes = floatTy.getIntOrFloatBitWidth() / 8;
    int64_t vectorWidth = kHVXWidthBytes / elemBytes;
    int64_t unrolledStep = vectorWidth * kUnrollFactor;
    auto vecTy = VectorType::get({vectorWidth}, elemTy);

    // Outer dim upper bounds (all except the reduction axis).
    SmallVector<Value> outerUppers;
    for (int64_t i = 0; i + 1 < rank; ++i) {
      if (memrefTy.isDynamicDim(i))
        outerUppers.push_back(memref::DimOp::create(rewriter, loc, input, i));
      else
        outerUppers.push_back(arith::ConstantIndexOp::create(
            rewriter, loc, memrefTy.getDimSize(i)));
    }

    // Reduction axis upper bound.
    int64_t axisStaticSize = memrefTy.getDimSize(rank - 1);
    Value axisUpper;
    if (ShapedType::isDynamic(axisStaticSize))
      axisUpper = memref::DimOp::create(rewriter, loc, input, rank - 1);
    else
      axisUpper = arith::ConstantIndexOp::create(rewriter, loc, axisStaticSize);

    // N as float for mean/variance computation.
    float normSizeValue = op.getNormSize().convertToFloat();
    Value nFloat = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elemTy, static_cast<double>(normSizeValue)));

    float epsValue = op.getEpsilon().convertToFloat();

    // Index and vector constants.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, unrolledStep);
    Value vwVal = arith::ConstantIndexOp::create(rewriter, loc, vectorWidth);

    Value zeroVec = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(vecTy,
                               APFloat::getZero(floatTy.getFloatSemantics())));
    Value oneF = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getFloatAttr(elemTy, 1.0));
    Value epsF = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemTy, epsValue));

    // Helper to compute indices at [outerIVs..., col + offset].
    auto indicesAt = [&](ValueRange outerIVs, Value col,
                         Value offset) -> SmallVector<Value> {
      SmallVector<Value> idx(outerIVs.begin(), outerIVs.end());
      if (offset)
        idx.push_back(arith::AddIOp::create(rewriter, loc, col, offset));
      else
        idx.push_back(col);
      return idx;
    };

    // Emit the layer norm body for a single row (given outer IVs).
    auto emitLayerNormBody = [&](ValueRange outerIVs) {
      // Loop 1: acc += x[i]  (sum for mean)
      auto meanLoop = scf::ForOp::create(rewriter, loc, zero, axisUpper, step,
                                         ValueRange{zeroVec});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(meanLoop.getBody());
        Value col = meanLoop.getBody()->getArgument(0);
        Value acc = meanLoop.getBody()->getArgument(1);
        auto idx0 = indicesAt(outerIVs, col, nullptr);
        auto idx1 = indicesAt(outerIVs, col, vwVal);

        Value v0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
        Value v1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);

        Value sum01 = arith::AddFOp::create(rewriter, loc, v0, v1, kFMFReassoc);
        Value newAcc =
            arith::AddFOp::create(rewriter, loc, acc, sum01, kFMFReassoc);
        scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
      }

      // Horizontal reduce -> mean scalar -> broadcast.
      Value meanSumVec = meanLoop.getResult(0);
      Value meanSumScalar = vector::ReductionOp::create(
          rewriter, loc, vector::CombiningKind::ADD, meanSumVec);
      setFastMath(meanSumScalar, kFMFReassoc);
      Value mean = arith::DivFOp::create(rewriter, loc, meanSumScalar, nFloat,
                                         kFMFReassoc);
      Value meanBcast = vector::BroadcastOp::create(rewriter, loc, vecTy, mean);

      // Loop 2: acc += (x[i] - mean)^2  (sum for variance)
      auto varLoop = scf::ForOp::create(rewriter, loc, zero, axisUpper, step,
                                        ValueRange{zeroVec});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(varLoop.getBody());
        Value col = varLoop.getBody()->getArgument(0);
        Value acc = varLoop.getBody()->getArgument(1);
        auto idx0 = indicesAt(outerIVs, col, nullptr);
        auto idx1 = indicesAt(outerIVs, col, vwVal);

        Value v0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
        Value c0 =
            arith::SubFOp::create(rewriter, loc, v0, meanBcast, kFMFReassoc);
        Value sq0 = arith::MulFOp::create(rewriter, loc, c0, c0, kFMFReassoc);

        Value v1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
        Value c1 =
            arith::SubFOp::create(rewriter, loc, v1, meanBcast, kFMFReassoc);
        Value sq1 = arith::MulFOp::create(rewriter, loc, c1, c1, kFMFReassoc);

        Value sum01 =
            arith::AddFOp::create(rewriter, loc, sq0, sq1, kFMFReassoc);
        Value newAcc =
            arith::AddFOp::create(rewriter, loc, acc, sum01, kFMFReassoc);
        scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
      }

      // Horizontal reduce -> variance -> rstd scalar -> broadcast.
      Value varSumVec = varLoop.getResult(0);
      Value varSumScalar = vector::ReductionOp::create(
          rewriter, loc, vector::CombiningKind::ADD, varSumVec);
      setFastMath(varSumScalar, kFMFReassoc);
      Value variance = arith::DivFOp::create(rewriter, loc, varSumScalar,
                                             nFloat, kFMFReassoc);
      Value shifted =
          arith::AddFOp::create(rewriter, loc, variance, epsF, kFMFReassoc);
      Value sqrtVal = math::SqrtOp::create(rewriter, loc, shifted);
      setFastMath(sqrtVal, kFMFReassoc);
      Value rstd =
          arith::DivFOp::create(rewriter, loc, oneF, sqrtVal, kFMFReassoc);
      Value rstdBcast = vector::BroadcastOp::create(rewriter, loc, vecTy, rstd);

      // Side stores: write mean and rstd to output memrefs if present.
      // When outerIVs don't cover all dims (e.g., 1-D input with rank-1
      // side memref from reinterpret_cast), pad with zero indices.
      auto sideStoreIndices = [&](Value memref) -> SmallVector<Value> {
        SmallVector<Value> indices(outerIVs.begin(), outerIVs.end());
        auto memrefTy = cast<MemRefType>(memref.getType());
        while (static_cast<int64_t>(indices.size()) < memrefTy.getRank())
          indices.push_back(zero);
        return indices;
      };
      if (meanOut)
        memref::StoreOp::create(rewriter, loc, mean, meanOut,
                                sideStoreIndices(meanOut));
      if (rstdOut)
        memref::StoreOp::create(rewriter, loc, rstd, rstdOut,
                                sideStoreIndices(rstdOut));

      // Loop 3: out[i] = (x[i] - mean) * rstd * w[i] + b[i]
      auto normLoop = scf::ForOp::create(rewriter, loc, zero, axisUpper, step);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(normLoop.getBody());
        Value col = normLoop.getBody()->getArgument(0);
        auto idx0 = indicesAt(outerIVs, col, nullptr);
        auto idx1 = indicesAt(outerIVs, col, vwVal);

        // Chunk 0.
        Value x0 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx0);
        Value w0 = vector::LoadOp::create(rewriter, loc, vecTy, weights, idx0);
        Value b0 = vector::LoadOp::create(rewriter, loc, vecTy, bias, idx0);
        Value centered0 =
            arith::SubFOp::create(rewriter, loc, x0, meanBcast, kFMFReassoc);
        Value normed0 = arith::MulFOp::create(rewriter, loc, centered0,
                                              rstdBcast, kFMFReassoc);
        Value weighted0 =
            arith::MulFOp::create(rewriter, loc, normed0, w0, kFMFReassoc);
        Value out0 =
            arith::AddFOp::create(rewriter, loc, weighted0, b0, kFMFReassoc);
        vector::StoreOp::create(rewriter, loc, out0, output, idx0);

        // Chunk 1.
        Value x1 = vector::LoadOp::create(rewriter, loc, vecTy, input, idx1);
        Value w1 = vector::LoadOp::create(rewriter, loc, vecTy, weights, idx1);
        Value b1 = vector::LoadOp::create(rewriter, loc, vecTy, bias, idx1);
        Value centered1 =
            arith::SubFOp::create(rewriter, loc, x1, meanBcast, kFMFReassoc);
        Value normed1 = arith::MulFOp::create(rewriter, loc, centered1,
                                              rstdBcast, kFMFReassoc);
        Value weighted1 =
            arith::MulFOp::create(rewriter, loc, normed1, w1, kFMFReassoc);
        Value out1 =
            arith::AddFOp::create(rewriter, loc, weighted1, b1, kFMFReassoc);
        vector::StoreOp::create(rewriter, loc, out1, output, idx1);
      }
    };

    // Nest scf.for loops over outer dimensions, emit body at innermost level.
    SmallVector<Value> ivs;
    std::function<void(unsigned)> buildOuter = [&](unsigned d) {
      if (d == outerUppers.size()) {
        emitLayerNormBody(ivs);
        return;
      }
      auto loop = scf::ForOp::create(rewriter, loc, zero, outerUppers[d], one);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      buildOuter(d + 1);
      rewriter.setInsertionPointAfter(loop);
      ivs.pop_back();
    };
    buildOuter(0);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerCompositePass
    : public ::impl::LowerCompositeBase<LowerCompositePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, composite::CompositeDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSoftmaxPattern>(&getContext());
    patterns.add<LowerRmsNormPattern>(&getContext());
    patterns.add<LowerLayerNormPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createLowerCompositePass() {
  return std::make_unique<LowerCompositePass>();
}
