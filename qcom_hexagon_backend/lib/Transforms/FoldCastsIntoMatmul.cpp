//===- FoldCastsIntoMatmul.cpp - Fold type casts into matmul -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//===----------------------------------------------------------------------===//
//
// This pass folds FP16→FP32 extension casts into linalg.matmul operations,
// enabling direct FP16 input with FP32 output matmuls.
//
// Before:
//   %lhs_fp32 = linalg.generic { arith.extf %lhs_fp16 : f16 to f32 }
//   %rhs_fp32 = linalg.generic { arith.extf %rhs_fp16 : f16 to f32 }
//   %result = linalg.matmul ins(%lhs_fp32, %rhs_fp32) outs(%out_fp32)
//
// After:
//   %result = linalg.matmul ins(%lhs_fp16, %rhs_fp16) outs(%out_fp32)
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fold-casts-into-matmul"

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Check if a generic op is a pure element-wise type extension cast.
/// Returns the source value if it's a cast, nullptr otherwise.
static Value getExtensionCastSource(linalg::GenericOp genericOp,
                                    Type sourceType, Type targetType) {
  if (!genericOp.hasPureTensorSemantics())
    return nullptr;

  // Must be element-wise (all identity maps) with one input and output
  if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return nullptr;

  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return nullptr;

  // Check the body performs a type extension cast
  Block &body = genericOp.getRegion().front();
  if (body.getOperations().size() != 2) // cast + yield
    return nullptr;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.back());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return nullptr;

  auto extfOp = yieldOp.getOperand(0).getDefiningOp<arith::ExtFOp>();
  if (!extfOp)
    return nullptr;

  if (extfOp.getIn().getType() != sourceType ||
      extfOp.getOut().getType() != targetType)
    return nullptr;

  auto blockArg = dyn_cast<BlockArgument>(extfOp.getIn());
  if (!blockArg || blockArg.getArgNumber() != 0)
    return nullptr;

  return genericOp.getDpsInputOperand(0)->get();
}

/// Pattern to fold FP16→FP32 casts into matmul operations.
struct FoldCastsIntoMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Checking matmul: " << matmulOp << "\n");

    // Get matmul operands
    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    Value out = matmulOp.getOutputs()[0];

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto outType = dyn_cast<RankedTensorType>(out.getType());

    if (!lhsType || !rhsType || !outType)
      return failure();

    // Check if output is FP32
    auto outElemType = outType.getElementType();
    if (!outElemType.isF32())
      return failure();

    // Check if inputs are FP32 (candidates for folding)
    auto lhsElemType = lhsType.getElementType();
    auto rhsElemType = rhsType.getElementType();

    if (!lhsElemType.isF32() || !rhsElemType.isF32())
      return failure();

    // Try to find FP16→FP32 cast producers
    auto lhsGeneric = lhs.getDefiningOp<linalg::GenericOp>();
    auto rhsGeneric = rhs.getDefiningOp<linalg::GenericOp>();

    if (!lhsGeneric && !rhsGeneric)
      return failure();

    auto f16Type = rewriter.getF16Type();
    auto f32Type = rewriter.getF32Type();

    // Get FP16 sources
    Value lhsFP16 = lhsGeneric
                        ? getExtensionCastSource(lhsGeneric, f16Type, f32Type)
                        : nullptr;
    Value rhsFP16 = rhsGeneric
                        ? getExtensionCastSource(rhsGeneric, f16Type, f32Type)
                        : nullptr;

    // If neither is a cast, nothing to do
    if (!lhsFP16 && !rhsFP16)
      return failure();

    // Use FP16 source if available, otherwise keep original
    Value newLhs = lhsFP16 ? lhsFP16 : lhs;
    Value newRhs = rhsFP16 ? rhsFP16 : rhs;

    LLVM_DEBUG({
      llvm::dbgs() << "Folding casts into matmul:\n";
      if (lhsFP16)
        llvm::dbgs() << "  LHS: FP16 source found\n";
      if (rhsFP16)
        llvm::dbgs() << "  RHS: FP16 source found\n";
    });

    // Create new matmul with FP16 inputs and FP32 output
    auto newMatmul = linalg::MatmulOp::create(
        rewriter, matmulOp.getLoc(), matmulOp.getResultTypes(),
        ValueRange{newLhs, newRhs}, ValueRange{out});

    rewriter.replaceOp(matmulOp, newMatmul.getResults());

    return success();
  }
};

struct FoldCastsIntoMatmulPass
    : public PassWrapper<FoldCastsIntoMatmulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldCastsIntoMatmulPass)

  StringRef getArgument() const final { return "fold-casts-into-matmul"; }

  StringRef getDescription() const final {
    return "Fold FP16→FP32 casts into linalg.matmul operations";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    LLVM_DEBUG(llvm::dbgs() << "=== Running FoldCastsIntoMatmulPass ===\n");

    RewritePatternSet patterns(ctx);
    patterns.add<FoldCastsIntoMatmul>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Clean up dead cast operations
    func.walk([](linalg::GenericOp genericOp) {
      if (genericOp.use_empty()) {
        genericOp.erase();
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "=== FoldCastsIntoMatmulPass Complete ===\n");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createFoldCastsIntoMatmulPass() {
  return std::make_unique<FoldCastsIntoMatmulPass>();
}

} // namespace hexagon
} // namespace mlir
