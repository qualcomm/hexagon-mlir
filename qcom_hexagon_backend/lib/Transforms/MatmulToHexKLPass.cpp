//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to transform linalg::MatmulOp to hexkl ops.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
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

#define DEBUG_TYPE "-matmul-to-hexkl"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToHexKL(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];
    rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
    return success();
  }
};

void populateMatmulToHexKLPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToHexKL>(patterns.getContext());
}

struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMatmulToHexKLPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createMatmulToHexKLPass() {
  return std::make_unique<MatmulToHexKLPass>();
}
