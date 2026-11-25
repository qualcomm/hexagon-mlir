//===- ThreadDistribution.cpp - Thread-level parallelism utilities --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/ThreadDistribution.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "thread-distribution"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::utils;

bool hexagon::computeTLPTilingOptions(IRRewriter &rewriter,
                                      ArrayRef<IteratorType> iteratorTypes,
                                      ArrayRef<int64_t> staticRanges,
                                      int64_t polytopeSize, int blockSize,
                                      int numThreads,
                                      scf::SCFTilingOptions &tilingOptions) {

  int64_t numLoops = iteratorTypes.size();
  assert(numLoops == static_cast<int64_t>(staticRanges.size()) &&
         "iteratorTypes and staticRanges must have the same size");

  // Count parallel loops.
  int64_t numParallelLoops = 0;
  for (auto type : iteratorTypes) {
    if (isParallelIterator(type))
      ++numParallelLoops;
  }

  // Case 0: no parallel loop.
  if (numParallelLoops == 0)
    return false;

  // Case 1: at least one of the loops is dynamic.
  for (int64_t i = 0; i < numLoops; ++i) {
    if (ShapedType::isDynamic(staticRanges[i]))
      return false;
  }

  // Divide range evenly across threads.
  auto computeThreadSize = [&](int64_t range) -> int64_t {
    return range <= numThreads ? 1 : (range / numThreads);
  };

  // To begin with, no interchange, single thread.
  SmallVector<int64_t> threadSizes(numLoops, 0);
  SmallVector<int64_t> interchangeVector(numLoops, 0);
  std::iota(interchangeVector.begin(), interchangeVector.end(), 0);

  // Make the parallel loop at `loopIndex` the threading loop by switching
  // it to outermost and splitting it into `threadSize` steps.
  auto configure = [&](size_t loopIndex, int64_t threadSize) {
    threadSizes[0] = threadSize;
    std::swap(interchangeVector[0], interchangeVector[loopIndex]);
    std::swap(threadSizes[0], threadSizes[loopIndex]);

    SmallVector<OpFoldResult> threadSizesOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), threadSizes);

    tilingOptions = scf::SCFTilingOptions();
    tilingOptions.setTileSizes(threadSizesOfr)
        .setInterchange(interchangeVector);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  };

  // User-specified blockSize for testing overrides other considerations.
  if (blockSize != 0) {
    configure(0, blockSize);
    return true;
  }

  // Case 2: is work large enough for multi-threading.
  if (polytopeSize < kMinPolytopeSize)
    return false;

  // Case 3: single parallel loop.
  if (numParallelLoops == 1) {
    for (int64_t i = 0; i < numLoops; ++i) {
      if (isParallelIterator(iteratorTypes[i])) {
        configure(i, computeThreadSize(staticRanges[i]));
        return true;
      }
    }
    llvm_unreachable("at least one parallel loop exists");
  }

  // Case N: multiple parallel loops.
  using DepthAndRange = std::pair<size_t, int64_t>;
  std::vector<DepthAndRange> dars;
  dars.reserve(numParallelLoops);
  for (int64_t i = 0; i < numLoops; ++i) {
    if (isParallelIterator(iteratorTypes[i]))
      dars.push_back(DepthAndRange{static_cast<size_t>(i), staticRanges[i]});
  }
  assert(dars.size() > 1 && "expected multiple parallel loops at this point");

  // Prefer the outermost parallel loop that can occupy every thread.
  for (const auto &[loopIndex, range] : dars) {
    if (range >= numThreads) {
      configure(loopIndex, computeThreadSize(range));
      return true;
    }
  }

  // No parallel dimension can fill the whole pool. Use the largest
  // non-unit dimension to maximize available parallelism.
  auto largest =
      std::max_element(dars.begin(), dars.end(),
                       [](const DepthAndRange &lhs, const DepthAndRange &rhs) {
                         return lhs.second < rhs.second;
                       });
  if (largest->second == 1)
    return false;

  configure(largest->first, computeThreadSize(largest->second));
  return true;
}
