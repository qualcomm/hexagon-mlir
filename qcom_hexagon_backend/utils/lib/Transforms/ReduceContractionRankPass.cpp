//===--- ReduceContractionRankPass.cpp -- Reduce rank of linalg ops --- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper pass to reduce rank of named contraction ops
// with unit extent dimension.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct ReduceContractionRankPass
    : public PassWrapper<ReduceContractionRankPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceContractionRankPass)

  StringRef getArgument() const override { return "reduce-contraction-rank"; }
  StringRef getDescription() const override {
    return "Reduce the rank of named contraction ops with unit extent dims "
           "(e.g., batch_matmul with batch=1 -> matmul; matvec with unit M -> "
           "dot)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(func.getContext());
    linalg::populateContractionOpRankReducingPatterns(patterns);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hexagon::createReduceContractionRankPass() {
  return std::make_unique<ReduceContractionRankPass>();
}
