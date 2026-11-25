//===- FlattenZeroLoopLinalgPass.cpp - Flatten assertion wrappers ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_FLATTENZEROLOOPLINALG
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace mlir::hexagon {
namespace {

// Triton emits side-effect-only zero-loop linalg.generic wrappers around
// scalar assertions.  Their bodies may contain control flow.  Flattening the
// wrapper before SCF-to-control-flow conversion keeps that control flow out
// of a linalg region, whose body must have at most one block.
struct FlattenZeroLoopLinalgPass
    : public ::impl::FlattenZeroLoopLinalgBase<FlattenZeroLoopLinalgPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenZeroLoopLinalgPass)

  void runOnOperation() override {
    SmallVector<linalg::GenericOp> wrappers;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op.getNumLoops() == 0 && op.getNumDpsInputs() == 0 &&
          op.getNumDpsInits() == 0 && op->getNumResults() == 0 &&
          op.getBody()->getNumArguments() == 0)
        wrappers.push_back(op);
    });

    for (linalg::GenericOp wrapper : wrappers) {
      Operation *insertBefore = wrapper.getOperation();
      SmallVector<Operation *> bodyOps;
      for (Operation &bodyOp : wrapper.getBody()->without_terminator())
        bodyOps.push_back(&bodyOp);
      for (Operation *bodyOp : bodyOps)
        bodyOp->moveBefore(insertBefore);
      wrapper.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFlattenZeroLoopLinalgPass() {
  return std::make_unique<FlattenZeroLoopLinalgPass>();
}

} // namespace mlir::hexagon
