//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts linalg.matmul with FP16 inputs to hexkl.matmul.
//
// Macro mode: Uses hexkl_macro_* API (f16 output only, upcast to f32 if needed)
//             Requires constant weights (preprocessed at compile-time)
// Micro mode: Uses hexkl_micro_* API (more flexible I/O dtype, memory
// allocation)
//
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-to-hexkl"

using namespace mlir;
using namespace hexagon;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  bool isMacroMode;

  MatmulToHexKL(MLIRContext *ctx, bool macroMode)
      : OpRewritePattern(ctx), isMacroMode(macroMode) {}

  bool isConstantWeight(Value weight) const {
    return weight.getDefiningOp<arith::ConstantOp>() != nullptr;
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];

    auto aType = dyn_cast<RankedTensorType>(A.getType());
    auto bType = dyn_cast<RankedTensorType>(B.getType());
    auto cType = dyn_cast<RankedTensorType>(C.getType());

    if (!aType || !bType || !cType) {
      return rewriter.notifyMatchFailure(op, "not ranked tensor types");
    }

    if (aType.getRank() != 2 || bType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    if (llvm::any_of(aType.getShape(), ShapedType::isDynamic) ||
        llvm::any_of(bType.getShape(), ShapedType::isDynamic)) {
      return rewriter.notifyMatchFailure(op,
                                         "dynamic dimensions not supported");
    }

    Type aElemType = aType.getElementType();
    Type bElemType = bType.getElementType();
    Type cElemType = cType.getElementType();

    if (!aElemType.isF16() || !bElemType.isF16()) {
      return rewriter.notifyMatchFailure(op, "inputs must be f16");
    }

    if (!cElemType.isF16() && !cElemType.isF32()) {
      return rewriter.notifyMatchFailure(op, "output must be f16 or f32");
    }

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    if (isMacroMode) {
      // Macro mode requires constant weights
      if (!isConstantWeight(B)) {
        return rewriter.notifyMatchFailure(
            op, "macro mode requires constant weights (weight must be "
                "arith.constant)");
      }

      const int64_t MAX_N_ROW = 1600;
      const int64_t MAX_N_COL = 5120;
      const int64_t MAX_N_INNER = 76030;

      if (M > MAX_N_ROW) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " exceeds max " +
                    std::to_string(MAX_N_ROW));
      }

      if (N > MAX_N_COL) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " exceeds max " +
                    std::to_string(MAX_N_COL));
      }

      if (K > MAX_N_INNER) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " exceeds max " +
                    std::to_string(MAX_N_INNER));
      }

      const int64_t BLOCK_SIZE = 32;

      if ((M % BLOCK_SIZE) != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (N % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (K % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[" << DEBUG_TYPE
                   << "] Converting matmul to hexkl.matmul ("
                   << (isMacroMode ? "macro" : "micro") << " mode):\n";
      llvm::dbgs() << "  LHS: " << aType << " (M=" << M << ", K=" << K << ")\n";
      llvm::dbgs() << "  RHS: " << bType << " (K=" << K << ", N=" << N << ")\n";
      llvm::dbgs() << "  Output: " << cType << " (M=" << M << ", N=" << N
                   << ")\n";
      if (isMacroMode) {
        llvm::dbgs() << "  Weight is constant: "
                     << (isConstantWeight(B) ? "yes" : "no") << "\n";
      }
    });

    if (isMacroMode) {
      auto f16Shape = cType.getShape();
      auto f16Type = RankedTensorType::get(f16Shape, rewriter.getF16Type());
      Value f16Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                rewriter.getF16Type());

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, f16Type, A, B, f16Output);

      Value result = hexklMatmul->getResult(0);

      if (cElemType.isF32()) {
        Value f32Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                  rewriter.getF32Type());

        AffineMap identityMap = rewriter.getMultiDimIdentityMap(2);
        SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
        SmallVector<utils::IteratorType> iteratorTypes = {
            utils::IteratorType::parallel, utils::IteratorType::parallel};

        auto castOp = linalg::GenericOp::create(
            rewriter, loc, cType, result, f32Output, indexingMaps,
            iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
              Value f16Val = args[0];
              Value f32Val =
                  arith::ExtFOp::create(b, loc, rewriter.getF32Type(), f16Val);
              linalg::YieldOp::create(b, loc, f32Val);
            });

        result = castOp.getResult(0);
      }

      rewriter.replaceOp(op, result);

    } else {
      Value outputOperand =
          tensor::EmptyOp::create(rewriter, loc, cType.getShape(), cElemType);

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, cType, A, B, outputOperand);

      rewriter.replaceOp(op, hexklMatmul->getResult(0));
    }

    return success();
  }
};

struct MatmulToHexKLPass
    : public hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  using hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass>::MatmulToHexKLBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    bool isMacroMode = (mode == "macro");

    if (mode != "macro" && mode != "micro") {
      emitError(getOperation()->getLoc(),
                "Invalid mode '" + mode + "'. Must be 'macro' or 'micro'");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] Running MatmulToHexKLPass (" << mode
               << " mode)\n");

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToHexKL>(&getContext(), isMacroMode);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" << DEBUG_TYPE << "] Pattern application failed\n");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] MatmulToHexKLPass complete\n");
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createMatmulToHexKLPass(const MatmulToHexKLOptions &options) {
  return std::make_unique<MatmulToHexKLPass>(options);
}

} // namespace hexagon
} // namespace mlir
