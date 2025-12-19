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

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
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

constexpr int64_t MIN_POLYTOPE_SIZE = 65536;

/// Returns true if the number of computations in the input genericOp,
/// i.e. points in the enclosing iteration space polytope, exceeds
/// a threshhold. Assumes given genericOp has static size.
bool willThreadingBeUseful(linalg::GenericOp genericOp) {
  SmallVector<int64_t> ranges = genericOp.getStaticLoopRanges();
  assert(
      llvm::none_of(
          ranges, [](int64_t range) { return ShapedType::isDynamic(range); }) &&
      "expected static shaped loop range at this stage.");

  int64_t polytopeSize =
      std::accumulate(ranges.begin(), ranges.end(), 1,
                      [](int64_t acc, int64_t r) { return acc * r; });
  polytopeSize *= genericOp.getBody()->getOperations().size() / 2;
  return polytopeSize >= MIN_POLYTOPE_SIZE;
}

// Implements control function to exploit thread level parallelism.
// A generic may have none, one, or multiple parallel axis. These
// may be distributed across depth and may not be at outer layer.
// Some may be unitary - no real parallelism. Others may have large
// range and need partitioning into few evenly distributed work-threads.
bool rewriteForTLP(IRRewriter &rewriter, linalg::GenericOp op,
                   scf::SCFTilingOptions &tilingOptions, int blockSize,
                   int numThreads) {

  // Case 0: no parallel loop.
  if (op.getNumParallelLoops() == 0)
    return false;

  // Case 1 : at least one of the loops is dynamic.
  auto ranges = op.getStaticLoopRanges();
  auto iterTypes = op.getIteratorTypesArray();
  for (size_t i = 0; i < iterTypes.size(); i++) {
    if (ShapedType::isDynamic(ranges[i])) {
      return false;
    }
  }

  // utility to divide the range (work) evenly across threads.
  auto computeThreadSize = [&](int64_t range) -> int64_t {
    return range <= numThreads ? 1 : (range / numThreads);
  };

  // To begin with, `no-interchange`, single thread.
  SmallVector<int64_t> threadSizes(op.getNumLoops(), 0);
  std::vector<long int> interchangeVector(op.getNumLoops(), 0);
  std::iota(interchangeVector.begin(), interchangeVector.end(), 0);

  // utility to make parallel-loop at `loopIndex` the threading loop
  // by switcing as outer-most and splitting it into `threadSize` steps.
  auto configure = [&](size_t loopIndex, int64_t threadSize) {
    threadSizes[0] = threadSize;
    std::swap(interchangeVector[0], interchangeVector[loopIndex]);
    std::swap(threadSizes[0], threadSizes[loopIndex]);

    SmallVector<OpFoldResult> threadSizesOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), threadSizes);

    tilingOptions = scf::SCFTilingOptions(); // reset the options.
    tilingOptions.setTileSizes(threadSizesOfr)
        .setInterchange(interchangeVector);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  };

  // User-specified blockSize for testing purposes overrides
  // other considerations. Just use it blindly.
  if (blockSize != 0) {
    configure(0, blockSize);
    return true;
  }

  // Loops are static from this point.
  // Case 2 : is work large enough for multi-threading.
  if (!willThreadingBeUseful(op))
    return false;

  // Case 3: single parallel loop.
  if (op.getNumParallelLoops() == 1) {
    for (size_t i = 0; i < iterTypes.size(); i++) {
      if (isParallelIterator(iterTypes[i])) {
        configure(i, computeThreadSize(ranges[i]));
        return true;
      }
    }
    llvm_unreachable("at least one parallel loop exists");
  }

  // Case N: multiple parallel loops.
  //   Identify the parallel loops and select one of them.
  using DepthAndRange = std::pair<size_t, int64_t>;
  std::vector<DepthAndRange> dars;
  dars.reserve(op.getNumParallelLoops());
  for (size_t i = 0; i < iterTypes.size(); i++) {
    if (isParallelIterator(iterTypes[i])) {
      dars.push_back(DepthAndRange{i, ranges[i]});
    }
  }
  assert(dars.size() > 1 && "expected multiple parallel loops at this point");

  // We can get away without transposes if outer is parallel
  if (dars[0].first == 0 && dars[0].second != 1) {
    configure(0, computeThreadSize(ranges[0]));
    return true;
  }

  // Pick a parallel loop that has multi-threading opportunity.
  for (size_t i = 0; i < dars.size(); ++i) {
    if (dars[i].second != 1) {
      configure(dars[i].first, computeThreadSize(dars[i].second));
      return true;
    }
  }
  // all parallel loops are unitary. No need to multi-thread.
  // TODO: revisit parallel-reduction option.
  return false;
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
  // access the pass options.

  funcOp.walk([&](linalg::GenericOp op) {
    IRRewriter rewriter(op.getContext());
    scf::SCFTilingOptions tilingOptions;

    // Get config for TLP.
    if (!rewriteForTLP(rewriter, op, tilingOptions, blockSize,
                       preferredNumThreads))
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
