//===- SpecializeLinalgMatmulPass.cpp - Specialize linalg matmul generics -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_SPECIALIZELINALGMATMUL
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

Operation *specializeAsMatmul(linalg::GenericOp op, IRRewriter &rewriter) {
  // Attempt linalg specialization on a temporary clone. Needed to keep
  // the original linalg.generic when the match fails.
  rewriter.setInsertionPoint(op);
  Operation *clone = rewriter.insert(op->clone());
  auto cloneGeneric = cast<linalg::GenericOp>(clone);
  FailureOr<linalg::LinalgOp> specialized =
      linalg::specializeGenericOp(rewriter, cloneGeneric);
  // If we can't specialize, no match.
  if (failed(specialized)) {
    rewriter.eraseOp(clone);
    return nullptr;
  }
  // If we specialize to anything besides linalg.matmul, no match.
  if (!isa<linalg::MatmulOp>(specialized->getOperation())) {
    rewriter.eraseOp(specialized->getOperation());
    return nullptr;
  }
  return specialized->getOperation();
}

struct SpecializeLinalgMatmulPass
    : public ::impl::SpecializeLinalgMatmulBase<SpecializeLinalgMatmulPass> {

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<linalg::GenericOp> generics;
    getOperation().walk([&](linalg::GenericOp op) { generics.push_back(op); });

    for (linalg::GenericOp op : generics) {
      Operation *specialized = specializeAsMatmul(op, rewriter);
      if (!specialized)
        continue;
      rewriter.replaceOp(op, specialized->getResults());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createSpecializeLinalgMatmulPass() {
  return std::make_unique<SpecializeLinalgMatmulPass>();
}
