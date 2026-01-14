//===- ExpandMathOpsPass.cpp - expand math ops ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Use available math expand paterns.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "expand-math-ops"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_EXPANDMATHOPS
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct ExpandMathOpsPass : public ::impl::ExpandMathOpsBase<ExpandMathOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    mlir::populateExpandTanPattern(patterns);
    mlir::populateExpandSinhPattern(patterns);
    mlir::populateExpandCoshPattern(patterns);
    mlir::populateExpandTanhPattern(patterns);
    mlir::populateExpandAsinhPattern(patterns);
    mlir::populateExpandAcoshPattern(patterns);
    mlir::populateExpandAtanhPattern(patterns);
    mlir::populateExpandRsqrtPattern(patterns);
    mlir::populateExpandCtlzPattern(patterns);
    mlir::populateExpandFmaFPattern(patterns);
    mlir::populateExpandCeilFPattern(patterns);
    mlir::populateExpandExp2FPattern(patterns);
    mlir::populateExpandPowFPattern(patterns);
    mlir::populateExpandFPowIPattern(patterns);
    mlir::populateExpandRoundFPattern(patterns);
    mlir::populateExpandRoundEvenPattern(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createExpandMathOpsPass() {
  return std::make_unique<ExpandMathOpsPass>();
}
