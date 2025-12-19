//===-- RemoveMLProgram.cpp - remove ML program global (not needed) -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace hexagon;
using namespace ml_program;

#define GEN_PASS_DEF_REMOVEMLPROGRAMPASS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Pattern to match and remove 'ml_program.global' operations
struct RemoveMLProgramPattern : public RewritePattern {
  // Constructor for the pattern, specifying the operation to match
  RemoveMLProgramPattern(MLIRContext *context)
      : RewritePattern(GlobalOp::getOperationName(), 1, context) {}

  // Check if the operation matches the pattern.
  // If succesful, rewrite the matched operation by erasing it
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto result = success(isa<GlobalOp>(op));
    if (result.succeeded())
      rewriter.eraseOp(op);
    return result;
  }
};

// Pass to apply the RemoveMLProgramPattern
struct RemoveMLProgramPass
    : public ::impl::RemoveMLProgramPassBase<RemoveMLProgramPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MLProgramDialect>();
  }

  // Run the pass on the module
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveMLProgramPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

// Create an instance of the RemoveMLProgramPass
std::unique_ptr<OperationPass<ModuleOp>> hexagon::removeMLProgramPass() {
  return std::make_unique<RemoveMLProgramPass>();
}
