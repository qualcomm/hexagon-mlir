//===-- MatmulToConvPass.cpp - linalg.matmul to linalg.conv_2d_nhwc_fhwc --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to transform linalg::MatmulOp to linalg::Conv2DNhwcFhwcOp for HMX.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "-matmul-to-conv"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_MATMULTOCONV
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Helper function to find the largest factor from the list that divides the
// value Returns {factor, quotient} where value = factor * quotient
static std::pair<int64_t, int64_t> factorize(int64_t value,
                                             ArrayRef<int64_t> factors) {
  for (int64_t factor : factors) {
    if (value % factor == 0) {
      return {factor, value / factor};
    }
  }
  // Should never reach here if factors includes 1
  return {1, value};
}

// Helper function to validate element types for matmul to conv conversion
// Currently supports i8 and f16 datatypes
static LogicalResult validateElementTypes(Value A, Value B) {
  auto elemA = llvm::dyn_cast<RankedTensorType>(A.getType()).getElementType();
  auto elemB = llvm::dyn_cast<RankedTensorType>(B.getType()).getElementType();
  if ((elemA != elemB) || !(elemA.isInteger(8) || elemA.isF16()))
    return failure();
  return success();
}

// Helper function to reshape a tensor to a new shape
static Value reshapeTensor(PatternRewriter &rewriter, Location loc,
                           Value tensor, ArrayRef<int64_t> newShape,
                           ArrayRef<ReassociationIndices> resoc) {
  ShapedType type = cast<ShapedType>(tensor.getType());
  auto newType = RankedTensorType::get(newShape, type.getElementType());
  return tensor::ExpandShapeOp::create(rewriter, loc, newType, tensor, resoc);
}

// Helper function to create conv2d operation and collapse result
static Value
createConvAndCollapseResult(PatternRewriter &rewriter, Location loc,
                            Value AReshaped, Value BTReshaped, Value CReshaped,
                            ArrayRef<ReassociationIndices> collapseIndices) {
  auto inputs = {AReshaped, BTReshaped};
  auto outputs = {CReshaped};
  auto conv = linalg::Conv2DNhwcFhwcOp::create(rewriter, loc, inputs, outputs);
  return tensor::CollapseShapeOp::create(rewriter, loc, conv.getResult(0),
                                         collapseIndices);
}

// Helper function to create transpose operation for 2D tensor (K, N) -> (N, K)
static Value createTranspose2D(PatternRewriter &rewriter, Location loc,
                               Value tensor, int64_t dim0, int64_t dim1,
                               Type elementType) {
  SmallVector<Value> dynamicDims;
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, ArrayRef<int64_t>{dim1, dim0}, elementType, dynamicDims);
  return linalg::TransposeOp::create(rewriter, loc, tensor, empty,
                                     ArrayRef<int64_t>{1, 0})
      ->getResult(0);
}

// Helper function to collapse leading batch dimension of size 1
// (1, dim1, dim2, ...) -> (dim1, dim2, ...)
static Value collapseBatchDim(PatternRewriter &rewriter, Location loc,
                              Value tensor) {
  ShapedType type = cast<ShapedType>(tensor.getType());
  auto shape = type.getShape();

  // Verify first dimension is 1
  assert(shape[0] == 1 && "Expected batch dimension to be 1");

  // Create reassociation indices: collapse first dimension with second
  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0, 1}); // Collapse dims 0 and 1
  for (int64_t i = 2; i < static_cast<int64_t>(shape.size()); ++i) {
    reassociation.push_back({i});
  }

  return tensor::CollapseShapeOp::create(rewriter, loc, tensor, reassociation);
}

// Common structure for matmul conversion parameters
struct MatmulConversionParams {
  int64_t heightFactor;
  int64_t heightQuotient;
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t batch; // 1 for regular matmul, actual batch size for batch_matmul
};

// Helper function to perform the common matmul to conv2d transformation.
// Takes 2D tensors A(M, K), B(K, N), C(M, N) and converts to conv2d.
// Returns the result tensor with shape (M, N).
static Value convertMatmulToConv2D(PatternRewriter &rewriter, Location loc,
                                   Value A, Value B, Value C,
                                   const MatmulConversionParams &params,
                                   Type elementType) {
  // Create transpose operation for B: (K, N) -> (N, K)
  Value BTransposed =
      createTranspose2D(rewriter, loc, B, params.K, params.N, elementType);

  // Reshape A, BT, C using factorized dimensions
  // A: (M, K) -> (1, heightFactor, heightQuotient, K)
  auto AReshaped =
      reshapeTensor(rewriter, loc, A,
                    {1, params.heightFactor, params.heightQuotient, params.K},
                    {{0, 1, 2}, {3}});
  // BT: (N, K) -> (N, 1, 1, K)
  auto BTReshaped = reshapeTensor(rewriter, loc, BTransposed,
                                  {params.N, 1, 1, params.K}, {{0}, {1, 2, 3}});
  // C: (M, N) -> (1, heightFactor, heightQuotient, N)
  auto CReshaped =
      reshapeTensor(rewriter, loc, C,
                    {1, params.heightFactor, params.heightQuotient, params.N},
                    {{0, 1, 2}, {3}});

  // Create conv and collapse result: (1, heightFactor, heightQuotient, N) ->
  // (M, N)
  SmallVector<ReassociationIndices> collapseIndices = {{0, 1, 2}, {3}};
  return createConvAndCollapseResult(rewriter, loc, AReshaped, BTReshaped,
                                     CReshaped, collapseIndices);
}

// Helper function to compute conversion parameters
static std::optional<MatmulConversionParams>
computeConversionParams(ArrayRef<int64_t> AShape, ArrayRef<int64_t> BShape,
                        bool isBatchMatmul) {
  MatmulConversionParams params;

  if (isBatchMatmul) {
    // BatchMatmul: A(batch, M, K), B(batch, K, N)
    if (AShape.size() != 3 || BShape.size() != 3)
      return std::nullopt;

    // Check that batch dimensions match
    if (AShape[0] != BShape[0])
      return std::nullopt;

    // Check that inner dimensions match
    if (AShape[2] != BShape[1])
      return std::nullopt;

    params.batch = AShape[0];
    params.M = AShape[1];
    params.K = AShape[2];
    params.N = BShape[2];

    // Only support batch size = 1. For batch=1, the BatchMatmulToConv pattern
    // collapses the batch dimension and uses the same transformation as
    // MatmulToConv.
    if (params.batch != 1)
      return std::nullopt;
  } else {
    // Regular Matmul: A(M, K), B(K, N)
    if (AShape.size() != 2 || BShape.size() != 2)
      return std::nullopt;

    params.batch = 1;
    params.M = AShape[0];
    params.K = AShape[1];
    params.N = BShape[1];

    // Check that inner dimensions match
    if (AShape[1] != BShape[0])
      return std::nullopt;
  }

  // Factorize M as multiple of {4, 2, 1}
  SmallVector<int64_t> heightFactors = {4, 2, 1};
  auto [heightFactor, heightQuotient] = factorize(params.M, heightFactors);
  params.heightFactor = heightFactor;
  params.heightQuotient = heightQuotient;

  return params;
}

// RewriterPattern for converting linalg::MatmulOp to linalg::Conv2DNhwcFhwcOp.
// linalg::Conv2DNhwcFhwcOp can be mapped to HMX by later passes.
//
// Example:
//   %0 = linalg.matmul(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>)
//                              -> tensor<8x32xf32>
//
// is rewritten to:
//   %A_reshaped = tensor.expand_shape %A : tensor<8x16xf32> to
//                   tensor<1x8x1x16xf32>
//   %B_transposed = linalg.transpose %B : tensor<16x32xf32> to
//                     tensor<32x16xf32>
//   %B_reshaped = tensor.expand_shape %B_transposed : tensor<32x16xf32> to
//                   tensor<32x1x1x16xf32>
//   %result = linalg.conv_2d_nhwc_fhwc %A_reshaped, %B_reshaped :
//               tensor<1x8x1x16xf32>, tensor<32x1x1x16xf32> ->
//               tensor<1x8x1x32xf32>
//   %result = tensor.collapse_shape %result : tensor<1x8x1x32xf32> to
//               tensor<8x32xf32>
//
// To achieve this we let the TransposeMatmul patterns to calculate
// B_transposed and convert linalg.matmul to linalg.matmul_transpose_b:
//
//
// Step 1) matmul(A, B) -> matmul_transpose_b(A, B_transpose)
// Step 2) matmul_transpose_b(A, B_transpose) -> conv_2d_nhwc_fhwc(A_reshaped,
//                                                                 B_reshaped)
struct MatmulToConv final : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToConv(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];

    auto AType = cast<ShapedType>(A.getType());
    auto BType = cast<ShapedType>(B.getType());
    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    // Validate element types (i8 and f16 are supported)
    if (failed(validateElementTypes(A, B)))
      return failure();

    // Compute conversion parameters
    auto paramsOpt =
        computeConversionParams(AShape, BShape, /*isBatchMatmul=*/false);
    if (!paramsOpt)
      return failure();
    auto params = *paramsOpt;

    // Note: Removed strict alignment checks (kVectorHeight, kChannelAlignment,
    // and crouton width constraints) to enable more flexible dimension
    // factorization. The new approach uses factorize() to find suitable tiling
    // factors, allowing the pass to apply to a wider range of matmul shapes
    // rather than only those meeting strict crouton alignment requirements.

    // Use common helper to perform the matmul to conv2d transformation
    auto result = convertMatmulToConv2D(rewriter, loc, A, B, C, params,
                                        BType.getElementType());

    rewriter.replaceOp(op, result);
    return success();
  }
};

// RewriterPattern for converting linalg::BatchMatmulOp to
// linalg::Conv2DNhwcFhwcOp. Currently only supports batch size = 1.
// For batch=1, we collapse the batch dimension and use the same logic
// as MatmulToConv, then expand the result back to include the batch dimension.
struct BatchMatmulToConv final
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  BatchMatmulToConv(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];

    auto AType = cast<ShapedType>(A.getType());
    auto BType = cast<ShapedType>(B.getType());
    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    // Validate element types (i8 and f16 are supported)
    if (failed(validateElementTypes(A, B)))
      return failure();

    // Compute conversion parameters
    auto paramsOpt =
        computeConversionParams(AShape, BShape, /*isBatchMatmul=*/true);
    if (!paramsOpt)
      return failure();
    auto params = *paramsOpt;

    // Since we only support batch == 1, collapse the batch dimension
    // to simplify the transformation. This makes the logic identical
    // to MatmulToConv.
    // A: (1, M, K) -> (M, K)
    Value ACollapsed = collapseBatchDim(rewriter, loc, A);
    // B: (1, K, N) -> (K, N)
    Value BCollapsed = collapseBatchDim(rewriter, loc, B);
    // C: (1, M, N) -> (M, N)
    Value CCollapsed = collapseBatchDim(rewriter, loc, C);

    // Use common helper to perform the matmul to conv2d transformation
    auto result =
        convertMatmulToConv2D(rewriter, loc, ACollapsed, BCollapsed, CCollapsed,
                              params, BType.getElementType());

    // Expand batch dimension back: (M, N) -> (1, M, N)
    auto resultExpanded = reshapeTensor(rewriter, loc, result,
                                        {1, params.M, params.N}, {{0, 1}, {2}});

    rewriter.replaceOp(op, resultExpanded);
    return success();
  }
};

void populateMatmulToConvPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToConv, BatchMatmulToConv>(patterns.getContext());
}

struct MatmulToConvPass : public ::impl::MatmulToConvBase<MatmulToConvPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateMatmulToConvPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> hexagon::createMatmulToConvPass() {
  return std::make_unique<MatmulToConvPass>();
}
