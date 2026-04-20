//===- OptimizeExtfTruncfOpPass.cpp - Optimize extf/truncf operations ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===---------------------------------------------------------------------===//
//
// This pass optimizes redundant floating-point conversion operations:
//
// 1. Removes back-to-back extf/truncf that cancel each other
//    (e.g., truncf -> extf with matching types)
//
// 2. Replaces tensor.empty followed by extf/truncf chains with a new
//    tensor.empty of the target type, eliminating intermediate conversions
//
// 3. Replaces linalg.fill followed by extf/truncf chains with a new
//    linalg.fill with target type constant and output tensor
//
//===---------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "optimize-extf-truncf-op"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::func;
using namespace hexagon;

#define GEN_PASS_DEF_OPTIMIZEEXTFTRUNCFOP
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Cache to store created tensor.empty ops to avoid duplicates
// Key: (shape, target element type) - using shape instead of Operation* to
// avoid dangling pointers
// Value: new tensor.empty op with target type
using TensorEmptyCache =
    llvm::DenseMap<std::pair<SmallVector<int64_t>, Type>, Value>;

// Check if the linalg.generic consisting of extf/truncf operations.
bool isLinalgGenericWithUnary(
    Operation *op, llvm::function_ref<bool(Operation &)> isDesiredOp) {
  if (!isa<linalg::GenericOp>(op))
    return false;

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumResults() != 1)
    return false;

  // Check that there is exactly one region and has exactly one block
  if (genericOp->getRegions().size() != 1 ||
      genericOp.getRegion().getBlocks().size() != 1)
    return false;

  Block *body = genericOp.getBody();
  if (!body || body->getOperations().size() != 2)
    return false;

  auto &ops = body->getOperations();
  return isa<linalg::YieldOp>(ops.back()) && isDesiredOp(ops.front());
}

bool isLinalgGenericWithExtF(Operation *op) {
  return isLinalgGenericWithUnary(
      op, [](Operation &firstOp) { return isa<arith::ExtFOp>(firstOp); });
}

bool isLinalgGenericWithTruncF(Operation *op) {
  return isLinalgGenericWithUnary(
      op, [](Operation &firstOp) { return isa<arith::TruncFOp>(firstOp); });
}

// Generic function to trace back through extf/truncf chains to find target op
template <typename TargetOpType>
Operation *traceThroughConversions(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (isa<TargetOpType>(defOp))
    return defOp;

  // If it's linalg.generic with extf/truncf, trace through it
  if (isLinalgGenericWithExtF(defOp) || isLinalgGenericWithTruncF(defOp))
    return traceThroughConversions<TargetOpType>(defOp->getOperand(0));

  return nullptr;
}

Operation *traceThroughConversionsToTensorEmpty(Value value) {
  return traceThroughConversions<tensor::EmptyOp>(value);
}

Operation *traceThroughConversionsToLinalgFill(Value value) {
  return traceThroughConversions<linalg::FillOp>(value);
}

// If there is an already existing tensor.empty op with the TargetDataType
// and the target shape, return it from the cache. Otherwise, create a new
// tensor.empty op with the target element type and store it in the cache.
Value getOrCreateTensorEmpty(Operation *originalEmpty, Type targetElemType,
                             PatternRewriter &rewriter,
                             TensorEmptyCache &cache) {
  auto emptyOp = cast<tensor::EmptyOp>(originalEmpty);
  auto originalType = cast<RankedTensorType>(emptyOp.getType());
  // Create cache key using shape (stable) instead of Operation* (can be
  // deleted)
  SmallVector<int64_t> shape(originalType.getShape().begin(),
                             originalType.getShape().end());
  auto cacheKey = std::make_pair(shape, targetElemType);
  // Check cache first
  if (cache.count(cacheKey)) {
    return cache[cacheKey];
  }

  // Create new tensor.empty with target element type
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(originalEmpty);

  Value newEmpty = tensor::EmptyOp::create(
      rewriter, emptyOp.getLoc(), originalType.getShape(), targetElemType);
  cache[cacheKey] = newEmpty;
  return newEmpty;
}

Value createConstantOfTargetTypeForFillOp(Operation *originalFill,
                                          Type targetElemType,
                                          Value originalConstant,
                                          PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(originalFill);

  if (auto constOp = originalConstant.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      double val = floatAttr.getValueAsDouble();
      auto targetAttr = FloatAttr::get(targetElemType, val);
      return arith::ConstantOp::create(rewriter, originalFill->getLoc(),
                                       targetElemType, targetAttr);
    }
  }
  return Value();
}

Value createOutputTensorOfTargetTypeForFillOp(Operation *originalFill,
                                              Type targetElemType,
                                              Value outputTensor,
                                              PatternRewriter &rewriter,
                                              TensorEmptyCache &cache) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(originalFill);

  auto outputType = cast<RankedTensorType>(outputTensor.getType());
  if (auto emptyOp = outputTensor.getDefiningOp<tensor::EmptyOp>())
    return getOrCreateTensorEmpty(emptyOp, targetElemType, rewriter, cache);

  return Value();
}

// Helper function to create a new linalg.fill with target type :
// Original linalg.fill:
// %empty_f32 = tensor.empty() : tensor<64x64xf32>
// %fill_f32 = linalg.fill ins(%cst_f32 : f32) outs(%empty_f32 :
// tensor<64x64xf32>)
//
// New tensor.empty & tensor.fill is created with target element type:
// %empty_f16 = tensor.empty() : tensor<64x64xf16>
// %fill_f16 = linalg.fill ins(%cst_f16 : f16) outs(%empty_f16 :
// tensor<64x64xf16>)
//
// This returns Value() if the output tensor of linalg.fill is an op other than
// tensor.empty.
Value createLinalgFillWithTargetType(Operation *originalFill,
                                     Type targetElemType,
                                     PatternRewriter &rewriter,
                                     TensorEmptyCache &cache) {
  auto fillOp = cast<linalg::FillOp>(originalFill);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(originalFill);

  Value newConstant = createConstantOfTargetTypeForFillOp(
      originalFill, targetElemType, fillOp.getInputs()[0], rewriter);
  Value newOutput = createOutputTensorOfTargetTypeForFillOp(
      originalFill, targetElemType, fillOp.getOutputs()[0], rewriter, cache);
  if (!newConstant || !newOutput)
    return Value();

  return linalg::FillOp::create(rewriter, originalFill->getLoc(),
                                ValueRange{newConstant}, ValueRange{newOutput})
      .getResult(0);
}

//===---------------------------------------------------------------------===//
// Pattern 1: Remove redundant arith.truncf -> arith.extf
//===---------------------------------------------------------------------===//
// Before :
// %truncf_out = arith.truncf(%in) : fp32 to fp16
// %extf_out = arith.extf(%truncf_out) : fp16 to fp32
// %result = some.op %extf_out : fp32  // consumer
//
// After :
// %result = some.op %in : fp32
//===---------------------------------------------------------------------===//
struct RemoveRedundantTruncFExtFPattern
    : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp extfOp,
                                PatternRewriter &rewriter) const override {
    auto truncfOp = extfOp.getIn().getDefiningOp<arith::TruncFOp>();
    if (!truncfOp)
      return failure();

    // Check if types match (truncf input type == extf output type)
    if (truncfOp.getIn().getType() != extfOp.getType())
      return failure();

    rewriter.replaceOp(extfOp, truncfOp.getIn());
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pattern 2: Remove redundant arith.extf -> arith.truncf
//===---------------------------------------------------------------------===//
// Before :
// %extf_out = arith.extf(%in) : fp16 to fp32
// %truncf_out = arith.truncf(%extf_out) : fp32 to fp16
// %result = some.op %truncf_out : fp16  // consumer
//
// After :
// %result = some.op %in : fp16
//===---------------------------------------------------------------------===//
struct RemoveRedundantExtFTruncFPattern
    : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp truncfOp,
                                PatternRewriter &rewriter) const override {
    auto extfOp = truncfOp.getIn().getDefiningOp<arith::ExtFOp>();
    if (!extfOp)
      return failure();

    // Check if types match (extf input type == truncf output type)
    if (extfOp.getIn().getType() != truncfOp.getType())
      return failure();

    rewriter.replaceOp(truncfOp, extfOp.getIn());
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pattern 3: Remove redundant linalg.generic(truncf) -> linalg.generic(extf)
//===---------------------------------------------------------------------===//
// If the output of linalg.generic(truncfOp)
// goes to the input of linalg.generic(extfOp) then,
// replace the uses of linalg.generic(extfOp) with the input of
// linalg.generic(truncfOp)
//
// Before :
// %in_extf = linalg.generic(truncfOp %in) : fp32 to fp16
// %out_extf = linalg.generic(extfOp %in_extf) : fp16 to fp32
// %result = some.op %out_extf : fp32  // consumer
//
// After :
// %result = some.op %in : fp32
//===---------------------------------------------------------------------===//
struct RemoveRedundantLinalgTruncFExtFPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isLinalgGenericWithExtF(genericOp))
      return failure();

    auto inputOp = genericOp.getOperand(0).getDefiningOp();
    if (!inputOp || !isLinalgGenericWithTruncF(inputOp))
      return failure();

    rewriter.replaceOp(genericOp, inputOp->getOperand(0));
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pattern 4: Remove redundant linalg.generic(extf) -> linalg.generic(truncf)
//===---------------------------------------------------------------------===//
// If the output of linalg.generic(extfOp)
// goes to the input of linalg.generic(truncfOp) then,
// replace the uses of linalg.generic(truncfOp) with the input of
// linalg.generic(extfOp)
//
// %in_truncf = linalg.generic(extfOp %in) : fp16 to fp32
// %out_truncf = linalg.generic(truncfOp %in_truncf) : fp32 to fp16
// %result = some.op %out_truncf : fp16  // consumer
//
// converts to :
// %result = some.op %in : fp16
//===---------------------------------------------------------------------===//
struct RemoveRedundantLinalgExtFTruncFPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isLinalgGenericWithTruncF(genericOp))
      return failure();

    auto inputOp = genericOp.getOperand(0).getDefiningOp();
    if (!inputOp || !isLinalgGenericWithExtF(inputOp))
      return failure();

    rewriter.replaceOp(genericOp, inputOp->getOperand(0));
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pattern 5: Optimize tensor.empty -> ... -> linalg.generic(extf/truncf)
//===---------------------------------------------------------------------===//
// Remove the intermediate extf/truncf ops which converts tensor.empty to
// target type, instead create a new tensor.empty of target type.
//
// Before :
// %empty_f32 = tensor.empty() : tensor<64x64xf32>
// %fp16_out = linalg.generic(truncf %empty_f32) : tensor<64x64xf32> to
// tensor<64x64xf16>
//
// After :
// %empty_f16 = tensor.empty() : tensor<64x64xf16>
//===---------------------------------------------------------------------===//
struct OptimizeTensorEmptyThroughConversionsPattern
    : public OpRewritePattern<linalg::GenericOp> {

  // Member variable to store the passed data
  TensorEmptyCache &cache;

  OptimizeTensorEmptyThroughConversionsPattern(MLIRContext *context,
                                               TensorEmptyCache &cache)
      : OpRewritePattern<linalg::GenericOp>(context), cache(cache) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isLinalgGenericWithExtF(genericOp) &&
        !isLinalgGenericWithTruncF(genericOp))
      return failure();

    // Trace back to tensor.empty
    Operation *emptyOp =
        traceThroughConversionsToTensorEmpty(genericOp.getOperand(0));
    if (!emptyOp)
      return failure();

    auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
    Type targetElemType = resultType.getElementType();
    Value newEmpty =
        getOrCreateTensorEmpty(emptyOp, targetElemType, rewriter, cache);

    if (!newEmpty)
      return failure();

    rewriter.replaceOp(genericOp, newEmpty);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pattern 6: Optimize linalg.fill -> ... -> linalg.generic(extf/truncf)
//===---------------------------------------------------------------------===//
// Remove the intermediate extf/truncf ops which converts linalg.fill output to
// target type, instead create a new linalg.fill with target type constant and
// output tensor.
//
// Before :
// %empty_f32 = tensor.empty() : tensor<64x64xf32>
// %cst_f32 = arith.constant 1.0 : f32
// %fill_f32 = linalg.fill ins(%cst_f32 : f32): tensor<64x64xf32>
// %fp16_out = linalg.generic(truncf %fill_f32) : tensor<64x64xf32 to
// tensor<64x64xf16>
//
// After :
// %empty_f16 = tensor.empty() : tensor<64x64xf16>
// %cst_f16 = arith.constant 1.0 : f16
// %fill_f16 = linalg.fill ins(%cst_f16 : f16)
//===---------------------------------------------------------------------===//
struct OptimizeLinalgFillThroughConversionsPattern
    : public OpRewritePattern<linalg::GenericOp> {

  // Member variable to store the passed data
  TensorEmptyCache &cache;

  OptimizeLinalgFillThroughConversionsPattern(MLIRContext *context,
                                              TensorEmptyCache &cache)
      : OpRewritePattern<linalg::GenericOp>(context), cache(cache) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isLinalgGenericWithExtF(genericOp) &&
        !isLinalgGenericWithTruncF(genericOp))
      return failure();

    // Trace back to linalg.fill
    Operation *fillOp =
        traceThroughConversionsToLinalgFill(genericOp.getOperand(0));
    if (!fillOp)
      return failure();

    auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
    Type targetElemType = resultType.getElementType();
    Value newFill =
        createLinalgFillWithTargetType(fillOp, targetElemType, rewriter, cache);
    if (!newFill)
      return failure();

    rewriter.replaceOp(genericOp, newFill);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pass Definition
//===---------------------------------------------------------------------===//

struct OptimizeExtfTruncfOpPass
    : public ::impl::OptimizeExtfTruncfOpBase<OptimizeExtfTruncfOpPass> {

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto funcOp = dyn_cast<func::FuncOp>(operation);
    if (!funcOp)
      return;

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Create cache once for the entire pass
    TensorEmptyCache cache;

    // Add all optimization patterns
    patterns.add<RemoveRedundantExtFTruncFPattern>(context);
    patterns.add<RemoveRedundantTruncFExtFPattern>(context);
    patterns.add<RemoveRedundantLinalgExtFTruncFPattern>(context);
    patterns.add<RemoveRedundantLinalgTruncFExtFPattern>(context);
    patterns.add<OptimizeTensorEmptyThroughConversionsPattern>(context, cache);
    patterns.add<OptimizeLinalgFillThroughConversionsPattern>(context, cache);
    // Apply patterns using greedy rewrite driver
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createOptimizeExtfTruncfOpPass() {
  return std::make_unique<OptimizeExtfTruncfOpPass>();
}
