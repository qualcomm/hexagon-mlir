//===- FormSCFThreadsPass.cpp : turn scf loop to multi-threaded ----------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass converts scf::for into a scf::forall so that `form-async-threads`
// pass subsequently turns it to a multi-threaded loop body. The responsibility
// for the correctness of the conversion is shared between user and compiler.
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/Passes.h"

#define DEBUG_TYPE "form-scf-threads"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_FORMSCFTHREADS
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Utility to convert `value` to `index` type.
Value convertToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  auto indexType = builder.getIndexType();
  auto indexValue =
      builder.create<mlir::arith::IndexCastOp>(loc, indexType, value);
  return indexValue;
}

/// Validate that the forOp has no iter args other than induction variable.
bool isValid(scf::ForOp forOp) {
  // Check that forOp has no iter args (only induction variable is allowed)
  if (!forOp.getInitArgs().empty())
    return false;
  return true;
}

void formSCFThreads(IRRewriter &rewriter, scf::ForOp forOp) {
  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);

  // do validity check.
  if (!isValid(forOp))
    return;

  // Extract bounds from the `scf::for` as `index` type.
  Value lowerBound = convertToIndex(rewriter, loc, forOp.getLowerBound());
  Value upperBound = convertToIndex(rewriter, loc, forOp.getUpperBound());
  Value step = convertToIndex(rewriter, loc, forOp.getStep());

  // convert to array of op-fold-results.
  SmallVector<OpFoldResult> lowerBounds{getAsOpFoldResult(lowerBound)};
  SmallVector<OpFoldResult> upperBounds{getAsOpFoldResult(upperBound)};
  SmallVector<OpFoldResult> steps{getAsOpFoldResult(step)};

  // create the `scf::forall` and move ops to it.
  auto forallOp = rewriter.create<scf::ForallOp>(
      loc, lowerBounds, upperBounds, steps, ValueRange(), std::nullopt);

  Block *forBody = forOp.getBody();
  Block *forallBody = forallOp.getBody();
  rewriter.setInsertionPointToStart(forallBody);

  // Create iv mapping from the old to the new one.
  IRMapping mapping;
  BlockArgument forallIV = forallBody->getArgument(0);
  BlockArgument forIV = forBody->getArgument(0);

  // In case `scf::forOp` was ixy etc type, cast to `index` type.
  if (isa<IntegerType>(forIV.getType())) {
    Value val = rewriter.create<mlir::arith::IndexCastOp>(loc, forIV.getType(),
                                                          forallIV);
    mapping.map(forIV, val);
  } else {
    mapping.map(forIV, forallIV);
  }

  // Clone from `scf::for` to `scf::forall`.
  for (auto &op : forBody->without_terminator())
    rewriter.clone(op, mapping);

  // As we assume no iter args, there are no results to replace.
  rewriter.eraseOp(forOp);
  return;
}

struct FormSCFThreadsPass
    : public ::impl::FormSCFThreadsBase<FormSCFThreadsPass> {
  void runOnOperation() override;
};

void FormSCFThreadsPass::runOnOperation() {
  auto funcOp = getOperation();
  funcOp.walk([&](scf::ForOp forOp) {
    IRRewriter rewriter(forOp.getContext());
    formSCFThreads(rewriter, forOp);
    return WalkResult::advance();
  });
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFormSCFThreadsPass() {
  return std::make_unique<FormSCFThreadsPass>();
}
