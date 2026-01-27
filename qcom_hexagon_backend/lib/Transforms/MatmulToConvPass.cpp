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

    constexpr int64_t kVectorHeight = 8;
    constexpr int64_t kChannelAlignment = 32;
    constexpr int64_t kTileWidthInt8 = 8;
    constexpr int64_t kTileWidthF16 = 4;

    auto AType = cast<ShapedType>(A.getType());
    auto BType = cast<ShapedType>(B.getType());
    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    // Currently i8 and f16 datatypes are supported.
    auto elemA = llvm::dyn_cast<RankedTensorType>(A.getType()).getElementType();
    auto elemB = llvm::dyn_cast<RankedTensorType>(B.getType()).getElementType();
    if ((elemA != elemB) || !(elemA.isInteger(8) || elemA.isF16()))
      return failure();

    // Constraints for checking if A & B can be reshapped without padding
    if (AShape[0] < kVectorHeight || AShape[0] % kVectorHeight != 0)
      return failure();
    // Checking if the channels and the number of filters are Channel aligned
    if (AShape[1] % kChannelAlignment != 0 ||
        BShape[1] % kChannelAlignment != 0)
      return failure();
    if (AShape[1] != BShape[0])
      return failure();

    // Tile Width based on the datatype
    auto cW = elemA.isInteger(8) ? kTileWidthInt8 : kTileWidthF16;

    if ((AShape[0] / 8) % cW)
      return failure();

    SmallVector<Value> dynamicDims;
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{BShape[1], BShape[0]}, BType.getElementType(),
        dynamicDims);
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, B, empty, ArrayRef<int64_t>{1, 0});

    // Reshape a tensor to new shape
    auto reshapeTensor = [&](Value tensor, ArrayRef<int64_t> newShape,
                             ArrayRef<ReassociationIndices> resoc) {
      ShapedType type = cast<ShapedType>(tensor.getType());
      auto newType = RankedTensorType::get(newShape, type.getElementType());
      Value reshaped =
          rewriter.create<tensor::ExpandShapeOp>(loc, newType, tensor, resoc);
      return reshaped;
    };

    // Reshape A, BT, C to their expanded shapes
    auto AReshaped = reshapeTensor(
        A, {1, kVectorHeight, AShape[0] / kVectorHeight, AShape[1]},
        {{0, 1, 2}, {3}});
    auto BTReshaped =
        reshapeTensor(transposeOp->getResult(0), {BShape[1], 1, 1, BShape[0]},
                      {{0}, {1, 2, 3}});
    auto CReshaped = reshapeTensor(
        C, {1, kVectorHeight, AShape[0] / kVectorHeight, BShape[1]},
        {{0, 1, 2}, {3}});

    // Create the conv_2d_nhwc_fhwc(AReshaped, BTReshaped, CReshaped) operation
    auto inputs = {AReshaped, BTReshaped};
    auto outputs = {CReshaped};
    auto conv = rewriter.create<linalg::Conv2DNhwcFhwcOp>(loc, inputs, outputs);

    // Convert the conv_2d_nhwc_fhwc result back to required type.
    SmallVector<ReassociationIndices> resocCBack = {{0, 1, 2}, {3}};
    auto result = rewriter.create<tensor::CollapseShapeOp>(
        loc, conv.getResult(0), resocCBack);

    rewriter.replaceOp(op, result);

    return success();
  }
};

void populateMatmulToConvPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToConv>(patterns.getContext());
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
