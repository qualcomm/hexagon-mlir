//===- FormVirtualThreadsPass.cpp : implement creation of threads   ------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass idenifies computations (e.g. linalg.generic) that can be tiled to
// scf::forall so that async dialect threads can be later created from them.
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/ThreadDistribution.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/Passes.h"

#define DEBUG_TYPE "form-virtual-threads"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_FORMVIRTUALTHREADS
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Compute a work-size estimate for the given GenericOp.
/// polytopeSize = product(all loop ranges) * (num_body_ops / 2).
static int64_t computePolytopeSize(linalg::GenericOp genericOp) {
  SmallVector<int64_t> ranges = genericOp.getStaticLoopRanges();
  int64_t size = std::accumulate(ranges.begin(), ranges.end(), int64_t{1},
                                 std::multiplies<int64_t>());
  size *= genericOp.getBody()->getOperations().size() / 2;
  return size;
}

struct FormVirtualThreadsPass
    : public ::impl::FormVirtualThreadsBase<FormVirtualThreadsPass> {
public:
  explicit FormVirtualThreadsPass(const FormVirtualThreadsOptions &options)
      : Base(options) {}
  void runOnOperation() override;
};

void FormVirtualThreadsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (preferredNumThreads <= 0) {
    funcOp.emitError() << "preferred-num-threads must be greater than zero";
    signalPassFailure();
    return;
  }

  funcOp.walk([&](linalg::GenericOp op) {
    IRRewriter rewriter(op.getContext());
    scf::SCFTilingOptions tilingOptions;

    auto iterTypes = op.getIteratorTypesArray();
    auto ranges = op.getStaticLoopRanges();
    int64_t polytopeSize = computePolytopeSize(op);

    if (!hexagon::computeTLPTilingOptions(rewriter, iterTypes, ranges,
                                          polytopeSize, blockSize,
                                          preferredNumThreads, tilingOptions))
      return WalkResult::advance();

    rewriter.setInsertionPoint(op);
    auto tileOp = cast<TilingInterface>(op.getOperation());

    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, tileOp, tilingOptions);
    if (failed(tiledResults))
      return WalkResult::advance();

    rewriter.replaceOp(op, tiledResults->replacements);
    return WalkResult::advance();
  });
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFormVirtualThreadsPass(
    const FormVirtualThreadsOptions &options) {
  return std::make_unique<FormVirtualThreadsPass>(options);
}
