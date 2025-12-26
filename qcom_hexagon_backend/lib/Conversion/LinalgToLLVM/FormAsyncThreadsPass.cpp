//===- FormAsyncThreadsPass.cpp :  Lower scf::forall to async threads ----====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass lowers virtual-threads to async.execute threads.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

#define DEBUG_TYPE "form-async-threads"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_FORMASYNCTHREADS
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {
LogicalResult formAsyncThreads(RewriterBase &rewriter, scf::ForallOp forallOp) {
  // If constraints are not met, resort to sequential execution.
  if (forallOp.getRank() != 1)
    return rewriter.notifyMatchFailure(
        forallOp, "expected the nesting depth of the forall op to be 1");

  if (!forallOp.getOutputs().empty())
    return rewriter.notifyMatchFailure(
        forallOp, "only fully bufferized scf.forall ops can be lowered");

  if (forallOp->getNumRegions() != 1 &&
          forallOp->getRegions().front().getBlocks().size() != 1 ||
      forallOp->getNumResults() != 0)
    return rewriter.notifyMatchFailure(
        forallOp, "scf::for all region/block not matching expectation");

  // Start construction ...
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  Location loc = forallOp.getLoc();

  // Convert mixed bounds and steps to SSA values.
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  // %group_id = async.create_group %num_threads : !async.group
  // For virtual threads lbs is '0' and range is affine map.
  Value nThreads = rewriter.create<arith::DivUIOp>(loc, ubs[0], steps[0]);
  auto groupId =
      rewriter.create<async::CreateGroupOp>(loc, nThreads).getResult();

  // Create a scf::for with an empty-body.
  scf::LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);

  SmallVector<Value> ivs = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  Block *forBody = loopNest.loops.back().getBody();

  // Create empty execute in for-body.
  rewriter.setInsertionPoint(forBody->getTerminator());
  auto executeOp = rewriter.create<async::ExecuteOp>(
      loc, TypeRange{}, ValueRange{}, ValueRange{});
  Value tokenId = executeOp.getResult(0);

  // Inline ops of forall-body into execute-body, before terminator.
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(
      forallOp.getBody(), executeOp.getBody(),
      executeOp.getBody()->getTerminator()->getIterator(), ivs);

  // scf::forall is broken IR at this stage.
  rewriter.eraseOp(forallOp);

  rewriter.setInsertionPoint(forBody->getTerminator());
  rewriter.create<async::AddToGroupOp>(loc, tokenId, groupId);

  // add await after the loop.
  rewriter.setInsertionPointAfter(loopNest.loops[0]);
  rewriter.create<async::AwaitAllOp>(loc, groupId);
  return success();
}

struct FormAsyncThreadsPass
    : public ::impl::FormAsyncThreadsBase<FormAsyncThreadsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<async::AsyncDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](scf::ForallOp op) {
      if (failed(formAsyncThreads(rewriter, op)))
        return signalPassFailure();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFormAsyncThreadsPass() {
  return std::make_unique<FormAsyncThreadsPass>();
}
