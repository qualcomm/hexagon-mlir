//===- EraseUnusedLinalgOperands.cpp - Enables EraseUnusedLinalgOperands --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Erase unused linalg operands.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "erase-unused-linalg-operands"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_ERASEUNUSEDLINALGOPERANDS
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct EraseUnusedLinalgOperandsPass
    : public ::impl::EraseUnusedLinalgOperandsBase<
          EraseUnusedLinalgOperandsPass> {
  void runOnOperation() override;
};

void EraseUnusedLinalgOperandsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
hexagon::createEraseUnusedLinalgOperands() {
  return std::make_unique<EraseUnusedLinalgOperandsPass>();
}
