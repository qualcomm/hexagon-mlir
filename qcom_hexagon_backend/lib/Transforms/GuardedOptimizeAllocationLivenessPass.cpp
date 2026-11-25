//===- GuardedOptimizeAllocationLivenessPass.cpp --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_GUARDEDOPTIMIZEALLOCATIONLIVENESS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct GuardedOptimizeAllocationLivenessPass
    : public ::impl::GuardedOptimizeAllocationLivenessBase<
          GuardedOptimizeAllocationLivenessPass> {
  void runOnOperation() override {
    auto func = getOperation();
    if (func->hasAttr("hexagon.uses_libdevice"))
      return;

    OpPassManager pm(func::FuncOp::getOperationName());
    pm.addPass(bufferization::createOptimizeAllocationLivenessPass());
    if (failed(runPipeline(pm, func)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createGuardedOptimizeAllocationLivenessPass() {
  return std::make_unique<GuardedOptimizeAllocationLivenessPass>();
}
