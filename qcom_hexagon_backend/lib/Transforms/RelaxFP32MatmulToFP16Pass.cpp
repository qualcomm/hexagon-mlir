//===- RelaxFP32MatmulToFP16Pass.cpp --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Rewrites fp32 linalg.matmul operations to consume fp16 inputs while
// preserving fp32 accumulation and fp32 output. Matmuls with a RHS-transposed
// indexing map are normalized with an explicit transpose so the existing HexKL
// lowering path can consume them as canonical linalg.matmul with fp16 inputs.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_RELAXFP32MATMULTOFP16
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

static Value createTensorF32ToF16Cast(PatternRewriter &rewriter, Location loc,
                                      Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto castType = RankedTensorType::get(
      inputType.getShape(), rewriter.getF16Type(), inputType.getEncoding());

  Value empty = tensor::EmptyOp::create(rewriter, loc, castType.getShape(),
                                        castType.getElementType());
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(inputType.getRank());
  SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);

  auto castOp = linalg::GenericOp::create(
      rewriter, loc, castType, input, empty, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
        Value castValue =
            arith::TruncFOp::create(b, bodyLoc, b.getF16Type(), args[0]);
        linalg::YieldOp::create(b, bodyLoc, castValue);
      });
  return castOp.getResult(0);
}

struct RelaxFP32Matmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "expected tensor semantics");

    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!lhsType || !rhsType || !initType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        initType.getRank() != 2 || resultType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 tensors");
    if (!lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !initType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "expected fp32 inputs/output");

    if (llvm::any_of(lhsType.getShape(), ShapedType::isDynamic) ||
        llvm::any_of(rhsType.getShape(), ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(op,
                                         "dynamic dimensions not supported");

    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    SmallVector<AffineMap> defaultIndexingMaps =
        linalg::MatmulOp::getDefaultIndexingMaps(op.getContext());
    bool transposeRhs = indexingMaps != defaultIndexingMaps;
    if (transposeRhs) {
      MLIRContext *ctx = op.getContext();
      defaultIndexingMaps[1] = AffineMap::get(
          3, 0, {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
      if (indexingMaps != defaultIndexingMaps)
        return rewriter.notifyMatchFailure(
            op, "expected default or RHS-transposed indexing maps");
    }

    Location loc = op.getLoc();
    Value lhsF16 = createTensorF32ToF16Cast(rewriter, loc, lhs);
    Value rhsF16 = createTensorF32ToF16Cast(rewriter, loc, rhs);
    if (transposeRhs) {
      Value rhsTransposeInit = tensor::EmptyOp::create(
          rewriter, loc,
          ArrayRef<int64_t>{rhsType.getShape()[1], rhsType.getShape()[0]},
          rewriter.getF16Type());
      rhsF16 =
          linalg::TransposeOp::create(rewriter, loc, rhsF16, rhsTransposeInit,
                                      ArrayRef<int64_t>{1, 0})
              ->getResult(0);
    }

    auto matmul =
        linalg::MatmulOp::create(rewriter, loc, op.getResultTypes(),
                                 ValueRange{lhsF16, rhsF16}, ValueRange{init});

    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

struct RelaxFP32MatmulToFP16Pass
    : public hexagon::impl::RelaxFP32MatmulToFP16Base<
          RelaxFP32MatmulToFP16Pass> {
  using hexagon::impl::RelaxFP32MatmulToFP16Base<
      RelaxFP32MatmulToFP16Pass>::RelaxFP32MatmulToFP16Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RelaxFP32Matmul>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<OperationPass<func::FuncOp>> createRelaxFP32MatmulToFP16Pass() {
  return std::make_unique<RelaxFP32MatmulToFP16Pass>();
}

} // namespace hexagon
} // namespace mlir