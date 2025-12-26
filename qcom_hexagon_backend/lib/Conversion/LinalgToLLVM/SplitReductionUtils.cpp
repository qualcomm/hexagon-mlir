//===- SplitReductionUtils.cpp - optimization for reduction loop ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file adds helper functions for split reduction
// which are pulled from upstream llvm/mlir repository.
//
// Split Reduction Implementation :
// The reduction loop present in the innermost dimension of linalg.generic op
// is converted into sequence of parallel loops followed by reduction.
// The parallel loops will give more opportunities for vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/SplitReductionUtils.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/AffineExpr.h>
#include <numeric>
#include <vector>
#define DEBUG_TYPE "split-reduction-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using namespace mlir::linalg;
using namespace mlir::transform;

#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<int64_t> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<int64_t>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Verify the tile size options are set in a consistent manner.
static LogicalResult verifyOptions(RewriterBase &rewriter, Location loc,
                                   const scf::SCFTilingOptions &options) {
  // Specifying number of threads is only supported on `scf.forall` op.
  if (options.numThreadsComputationFunction &&
      options.loopType != scf::SCFTilingOptions::LoopType::ForallOp) {
    return rewriter.notifyMatchFailure(
        loc, "number of threads can only by specified when loop type is "
             "set to use `scf.forall`");
  }

  // If specified, check that the interchange vector is a permutation.
  if (!options.interchangeVector.empty()) {
    if (!isPermutationVector(options.interchangeVector)) {
      return rewriter.notifyMatchFailure(
          loc, "invalid interchange vector, not a permutation of the entire "
               "iteration space");
    }
  }
  return success();
}

/// Method to instantiate the tile sizes and/or number of threads specified
/// by the user.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getUserTileSizesAndNumThreads(RewriterBase &rewriter, TilingInterface op,
                              ArrayRef<Range> iterationDomain,
                              const scf::SCFTilingOptions &options) {
  OpFoldResult zero = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> tileSizes, numThreads;
  size_t numLoops = iterationDomain.size();

  // Check whether the number of tiles to use is specified.
  if (options.numThreadsComputationFunction) {
    numThreads = options.numThreadsComputationFunction(rewriter, op);
    numThreads.resize(numLoops, zero);

    // If the number of tiles is also specified, use that.
    if (options.tileSizeComputationFunction) {
      tileSizes = options.tileSizeComputationFunction(rewriter, op);
      tileSizes.resize(numLoops, zero);
      return {tileSizes, numThreads};
    }

    // Compute the tile sizes from the iteration domain and number
    // of tiles as follows
    // - niters = ceilDiv(ub - lb, step)
    // - tileSize = ceilDiv(niters, numThreads)
    AffineExpr s0, s1, s2;
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    // TODO: The step here is assumed to be 1.
    AffineExpr numItersExpr = (s1 - s0);
    AffineExpr tileSizeExpr = numItersExpr.ceilDiv(s2);
    tileSizes.resize(numLoops, zero);
    for (auto [index, range, nt] :
         llvm::enumerate(iterationDomain, numThreads)) {
      if (isZeroInteger(nt))
        continue;

      tileSizes[index] = affine::makeComposedFoldedAffineApply(
          rewriter, op.getLoc(), tileSizeExpr, {range.offset, range.size, nt});
    }
    tileSizes.resize(numLoops, zero);
    return {tileSizes, numThreads};
  }

  // Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  assert(options.tileSizeComputationFunction &&
         "expected tile sizes to be specified");
  tileSizes = options.tileSizeComputationFunction(rewriter, op);
  tileSizes.resize(numLoops, zero);

  return {tileSizes, numThreads};
}

/// Checks if any of the tiled loops are not parallel.
static LogicalResult checkTileSizes(TilingInterface op,
                                    scf::SCFTilingOptions::LoopType loopType,
                                    ReductionTilingStrategy reductionStrategy,
                                    ArrayRef<OpFoldResult> tileSizes,
                                    ArrayRef<OpFoldResult> numThreads) {
  auto iterators = op.getLoopIteratorTypes();
  assert(iterators.size() == tileSizes.size() &&
         "expected as many tile size values as number of loops");
  assert((numThreads.empty() || (numThreads.size() == iterators.size())) &&
         "when specified, expected number of threads to use for each loop");

  bool isParallelTiling = false;
  for (auto [index, iterator, tileSize] :
       llvm::enumerate(iterators, tileSizes)) {
    if (!isConstantIntValue(tileSize, 0)) {
      isParallelTiling |= iterator == utils::IteratorType::parallel;
    }

    if (loopType == scf::SCFTilingOptions::LoopType::ForallOp &&
        reductionStrategy == ReductionTilingStrategy::FullReduction) {
      // If num threads is specified, check that it is greater than one only for
      // parallel dimensions.
      if (!numThreads.empty()) {
        if (std::optional<int64_t> constNumThreads =
                getConstantIntValue(numThreads[index])) {
          if (constNumThreads.value() > 1 &&
              iterator != utils::IteratorType::parallel) {
            op.emitWarning() << "tiling is not thread safe at axis #" << index;
          }
        }
        continue;
      }

      if (std::optional<int64_t> constTileSize =
              getConstantIntValue(tileSize)) {
        if (constTileSize.value() > 0 &&
            iterator != utils::IteratorType::parallel) {
          op.emitWarning() << "tiling is not thread safe at axis #" << index;
        }
      }
    }
  }

  if (reductionStrategy != ReductionTilingStrategy::FullReduction) {
    if (isParallelTiling) {
      return op->emitOpError("tiling parallel dimensions is not supported with "
                             "partial reduction tiling strategies");
    }
  }
  return success();
}

/// Get the reduction dims that are tiled. This accounts for reduction dims
/// that are specified as tiled, but the tile size is 0.
static SetVector<unsigned>
getSanitizedReductionDims(ArrayRef<OpFoldResult> tileSizes,
                          const scf::SCFTilingOptions &options) {
  SetVector<unsigned> reductionDims;
  for (auto dim : options.reductionDims) {
    if (isConstantIntValue(tileSizes[dim], 0))
      continue;
    reductionDims.insert(dim);
  }
  return reductionDims;
}

/// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  std::optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  std::optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  std::optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// Returns the bounded tile size given the current `offset`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - offset)`.
static OpFoldResult getBoundedTileSize(OpBuilder &b, Location loc,
                                       Range loopRange, OpFoldResult offset,
                                       OpFoldResult tileSize) {
  std::optional<int64_t> ts = getConstantIntValue(tileSize);
  if (ts && ts.value() == 1)
    return tileSize;

  if (tileDividesIterationDomain(
          Range{loopRange.offset, loopRange.size, tileSize}))
    return tileSize;

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable of the tiled
  // loop.
  AffineExpr s0, s1, d0;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0 - d0, s1}, b.getContext());
  Value size = getValueOrCreateConstantIndexOp(b, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{offset, size, tileSize});
}

/// Returns true if the maximum tile offset `tileSize * numThreads-1` is less
/// than `iterationSize`.
static bool canOmitTileOffsetInBoundsCheck(OpFoldResult tileSize,
                                           OpFoldResult numThreads,
                                           OpFoldResult iterationSize) {
  std::optional<int64_t> tileSizeConst = getConstantIntValue(tileSize);
  std::optional<int64_t> numThreadsConst = getConstantIntValue(numThreads);
  std::optional<int64_t> iterSizeConst = getConstantIntValue(iterationSize);
  if (!tileSizeConst || !numThreadsConst || !iterSizeConst)
    return false;
  return *tileSizeConst * (*numThreadsConst - 1) < *iterSizeConst;
}

/// Compute the `OpFoldResult`s that represents the multi-dimensional
/// `offset`s and `size`s of the tile of the iteration space that the
/// innermost loop body of the generated tiled loops corresponds to.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getTileOffsetAndSizes(RewriterBase &rewriter, Location loc,
                      ReductionTilingStrategy strategy, ValueRange ivs,
                      ArrayRef<Range> iterationDomain,
                      ArrayRef<OpFoldResult> tileSizes,
                      ArrayRef<OpFoldResult> numThreads,
                      const llvm::SetVector<unsigned> &reductionDims) {
  SmallVector<OpFoldResult> offsets, sizes;
  int materializedLoopNum = 0;

  if (!numThreads.empty()) {
    AffineExpr d0, d1, s0, s1;
    AffineExpr offsetExpr, residualTileSizeExpr;
    bindDims(rewriter.getContext(), d0, d1);
    bindSymbols(rewriter.getContext(), s0, s1);
    offsetExpr = d0 + d1 * s0;
    residualTileSizeExpr = s1 - (d0 + d1 * s0);

    for (auto [index, nt, tileSize, loopRange] :
         llvm::enumerate(numThreads, tileSizes, iterationDomain)) {

      // Non-tiled cases, set the offset and size to the
      // `loopRange.offset/size`.
      if (isZeroInteger(nt)) {
        offsets.push_back(loopRange.offset);
        sizes.push_back(loopRange.size);
        continue;
      }

      Value iv = ivs[materializedLoopNum++];
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          rewriter, loc, offsetExpr,
          ArrayRef<OpFoldResult>{loopRange.offset, iv, tileSize});
      OpFoldResult residualTileSize = affine::makeComposedFoldedAffineApply(
          rewriter, loc, residualTileSizeExpr,
          {loopRange.offset, nt, tileSize, loopRange.size});

      OpFoldResult size = tileSize;
      if (!isZeroInteger(residualTileSize)) {
        OpFoldResult sizeMinusOffsetPerThread =
            affine::makeComposedFoldedAffineApply(rewriter, loc, s0 - d0,
                                                  {offset, loopRange.size});
        size = affine::makeComposedFoldedAffineMin(
            rewriter, loc,
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            {sizeMinusOffsetPerThread, tileSize});
      }

      // Consider the case where the original loop was `[0, 100)`.
      // If number of threads are `7`, the tile size would be computed as
      // `ceilDiv(100, 7) = 15`. For the last thread (thread_id = 6)
      // - `offset = 0 + 6 * 15 = 105`
      // - `tileSize = min(15, 100 - 105) = -5`
      // To avoid negative tile sizes, we need to do a further
      // `nonNegativeTileSize = affine.max(0, tileSize)`.
      // This `max` can be avoided if
      //  `offset + tileSize * (numThreads - 1) < (ub - lb)`
      if (!canOmitTileOffsetInBoundsCheck(tileSize, nt, loopRange.size)) {
        AffineMap maxMap =
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
        size = affine::makeComposedFoldedAffineMax(
            rewriter, loc, maxMap, {rewriter.getIndexAttr(0), size});
      }

      offsets.push_back(offset);
      sizes.push_back(size);
    }
    return {offsets, sizes};
  } else {
    for (auto [tileSize, loopRange] :
         llvm::zip_equal(tileSizes, iterationDomain)) {

      // Non-tiled cases, set the offset and size to the
      // `loopRange.offset/size`.
      if (isZeroInteger(tileSize)) {
        offsets.push_back(loopRange.offset);
        sizes.push_back(loopRange.size);
        continue;
      }

      Value iv = ivs[materializedLoopNum++];
      OpFoldResult offset = getAsOpFoldResult(iv);
      offsets.push_back(offset);
      OpFoldResult size =
          getBoundedTileSize(rewriter, loc, loopRange, offset, tileSize);
      sizes.push_back(size);
    }
    return {offsets, sizes};
  }
}

/// Function to return the bounds of the loops to be generated.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<OpFoldResult>>
getLoopBounds(RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
              ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  for (auto [loopRange, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    // No loop if the tile size is 0.
    if (isZeroInteger(tileSize))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(tileSize);
  }
  return {lbs, ubs, steps};
}

/// A function that allows returning additional yielded values during
/// `yieldTiledValuesAndReplace`.
/// - `ivs` induction variable for the loop.
/// - `newBbArgs` basic block arguments corresponding to newly added iter_args.
/// - `tiledValues` the tiled values to return. Must be of same size as
///   `newbbArgs`, each element of this array is inserted into the corresponding
///   element in `newbbArgs`.
/// - `resultOffsets` is of the same size as `tiledValues` and represents
///   the offsets to use when inserting corresponding element from `tiledValues`
///   into the element from `newBbArgs`.
/// - `resultSizes` is of the same size as `tiledValues` and represents
///   the size of the corresponding element from `tiledValues` inserted into
///   the element from `newBbArgs`.
/// In case the method needs to return `failure()` the method is expected
/// to clean up any inserted operations.
using YieldTiledValuesFn = std::function<LogicalResult(
    RewriterBase &rewriter, Location loc, ValueRange ivs, ValueRange newBbArgs,
    SmallVector<Value> &tiledValues,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes)>;

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty())
    return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// Generate the tile-loop nest using `scf.for` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ValueRange destinationTensors,
    YieldTiledValuesFn yieldTiledValuesFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);

  SmallVector<OpFoldResult> lbs, ubs, steps;
  std::tie(lbs, ubs, steps) =
      getLoopBounds(rewriter, loc, loopRanges, tileSizes);
  SmallVector<Value> lbVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, lbs);
  SmallVector<Value> ubVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, ubs);
  SmallVector<Value> stepVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, steps);

  SmallVector<Value> ivs;
  for (auto [lb, ub, step] : llvm::zip_equal(lbVals, ubVals, stepVals)) {
    auto loop =
        scf::ForOp::create(rewriter, loc, lb, ub, step, destinationTensors,
                           [](OpBuilder &bodyBuilder, Location bodyLoc,
                              Value iv, ValueRange /*iterArgs*/) {});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToEnd(loop.getBody());
    destinationTensors = loop.getRegionIterArgs();
  }

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(yieldTiledValuesFn(rewriter, loc, ivs, destinationTensors,
                                tiledResults, resultOffsets, resultSizes))) {
    return rewriter.notifyMatchFailure(
        loc, "failed to generate inner tile loop body");
  }
  if (loops.empty())
    return success();

  assert(tiledResults.size() == destinationTensors.size() &&
         "Number of results of body should be equal to number of iter args");

  // 6. Yield all the results of the tiled operation.
  SmallVector<Value> yieldedValues;
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    auto insertSlice = tensor::InsertSliceOp::create(
        rewriter, loc, tiledValue, destinationTensor, resultOffset, resultSize,
        resultStride);
    yieldedValues.push_back(insertSlice);
  }
  scf::YieldOp::create(rewriter, loc, yieldedValues);

  // Add the scf.yield operations for all the outer loops.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(MutableArrayRef(loops).drop_back(),
                       MutableArrayRef(loops).drop_front())) {
    rewriter.setInsertionPointToEnd(
        cast<scf::ForOp>(outerLoop.getOperation()).getBody());
    scf::YieldOp::create(rewriter, outerLoop.getLoc(), innerLoop->getResults());
  }
  return success();
}

/// Generate the tile-loop nest using `scf.forall` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `mappingVector` is the mapping attributes to use for loop construction.
///   Can be empty.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForallOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> numThreads,
    ArrayRef<Attribute> mappingVector, ValueRange destinationTensors,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  scf::ForallOp forallOp;
  bool useNumThreads = !numThreads.empty();

  if (useNumThreads) {
    // Prune the zero numthreads.
    SmallVector<OpFoldResult> nonZeroNumThreads;
    for (auto nt : numThreads) {
      if (isZeroInteger(nt))
        continue;
      nonZeroNumThreads.push_back(nt);
    }
    forallOp = scf::ForallOp::create(rewriter, loc, nonZeroNumThreads,
                                     destinationTensors, mappingAttr);
  } else {
    SmallVector<OpFoldResult> lbs, ubs, steps;
    std::tie(lbs, ubs, steps) =
        getLoopBounds(rewriter, loc, loopRanges, tileSizes);
    forallOp = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                     destinationTensors, mappingAttr);
  }
  loops.push_back(forallOp);

  rewriter.setInsertionPoint(forallOp.getTerminator());
  destinationTensors = forallOp.getRegionOutArgs();

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(tiledBodyFn(rewriter, loc, forallOp.getInductionVars(),
                         destinationTensors, tiledResults, resultOffsets,
                         resultSizes)))
    return rewriter.notifyMatchFailure(loc, "failed to generate loop body");

  rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));

    tensor::ParallelInsertSliceOp::create(rewriter, loc, tiledValue,
                                          destinationTensor, resultOffset,
                                          resultSize, resultStride);
  }
  return success();
}

/// Generate the tile-loop nest using the loop construct specifed in `options`.
/// - `options`: Tiling options specified.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNest(
    RewriterBase &rewriter, Location loc,
    scf::SCFTilingOptions::LoopType loopType, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> numThreads,
    ValueRange destinationTensors, ArrayRef<Attribute> mappingVector,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  // If the tile sizes are all zero, no loops are generated. Just call the
  // callback function to handle untiled case.
  if (llvm::all_of(tileSizes, isZeroInteger)) {
    SmallVector<Value> tiledResults;
    SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
    return tiledBodyFn(rewriter, loc, ValueRange{}, destinationTensors,
                       tiledResults, resultOffsets, resultSizes);
  }
  if (loopType == scf::SCFTilingOptions::LoopType::ForOp) {
    return generateLoopNestUsingForOp(rewriter, loc, loopRanges, tileSizes,
                                      destinationTensors, tiledBodyFn, loops);
  }
  if (loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
    return generateLoopNestUsingForallOp(
        rewriter, loc, loopRanges, tileSizes, numThreads, mappingVector,
        destinationTensors, tiledBodyFn, loops);
  }
  return rewriter.notifyMatchFailure(loc, "unhandled loop type");
}

static FailureOr<SmallVector<Value>> createInitialTensorsForTiling(
    RewriterBase &rewriter, TilingInterface op,
    ReductionTilingStrategy reductionStrategy, ArrayRef<Range> iterationDomain,
    ArrayRef<OpFoldResult> numThreads, ArrayRef<OpFoldResult> tileSizes,
    const SetVector<unsigned> &reductionDims) {
  SmallVector<Value> initTensors;
  Location loc = op->getLoc();
  if (reductionStrategy == ReductionTilingStrategy::FullReduction) {
    if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, initTensors)))
      return failure();
    return initTensors;
  }

  auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
  if (!redOp) {
    return op->emitOpError(
        "PartialReductionOuterReduction tiling strategy is only supported for "
        "operations implementing PartialReductionOpInterface");
  }
  SmallVector<OpFoldResult> sizes(iterationDomain.size());
  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr sizeExpr = ((s0 - s1).ceilDiv(s2));
  AffineExpr divExpr = s0.ceilDiv(s1);
  for (auto [index, domain, tileSize] :
       llvm::enumerate(iterationDomain, tileSizes)) {
    if (!numThreads.empty()) {
      // Untiled case.
      if (isConstantIntValue(numThreads[index], 0)) {
        sizes[index] = affine::makeComposedFoldedAffineApply(
            rewriter, op.getLoc(), sizeExpr,
            {domain.size, domain.offset, domain.stride});
        continue;
      }
      sizes[index] = numThreads[index];
      continue;
    }

    // Non reduction dimensions/non-tiled dimensions.
    if (!reductionDims.contains(index) || isConstantIntValue(tileSize, 0)) {
      sizes[index] = affine::makeComposedFoldedAffineApply(
          rewriter, op.getLoc(), sizeExpr,
          {domain.size, domain.offset, domain.stride});
      continue;
    }

    if (reductionStrategy ==
        ReductionTilingStrategy::PartialReductionOuterReduction) {
      sizes[index] = tileSize;
      continue;
    }

    assert(reductionStrategy ==
           ReductionTilingStrategy::PartialReductionOuterParallel);
    OpFoldResult normalizedRange = affine::makeComposedFoldedAffineApply(
        rewriter, op.getLoc(), sizeExpr,
        {domain.size, domain.offset, domain.stride});
    sizes[index] = affine::makeComposedFoldedAffineApply(
        rewriter, op.getLoc(), divExpr, {normalizedRange, tileSize});
  }
  return redOp.generateInitialTensorForPartialReduction(rewriter, loc, sizes,
                                                        reductionDims);
}

/// For the case of `ReductionTilingStrategy::PartialReductionOuterParallel`
/// the `PartialReductionOpInterface` methods need the index of the parallel
/// split reduction being executed.
static SmallVector<OpFoldResult>
getSplitReductionIvs(RewriterBase &rewriter, Location loc,
                     ReductionTilingStrategy reductionStrategy, ValueRange ivs,
                     ArrayRef<OpFoldResult> numThreads,
                     ArrayRef<OpFoldResult> tileSizes,
                     const SetVector<unsigned> &reductionDims) {
  SmallVector<OpFoldResult> splitReductionIvs;
  splitReductionIvs.resize(reductionDims.size(), rewriter.getIndexAttr(0));
  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineExpr divExpr = s0.floorDiv(s1);
  int ivIndex = 0;
  if (reductionStrategy ==
      ReductionTilingStrategy::PartialReductionOuterParallel) {
    for (auto [index, reductionDim] : llvm::enumerate(reductionDims)) {
      if (!numThreads.empty()) {
        splitReductionIvs[index] = ivs[ivIndex++];
        continue;
      }
      splitReductionIvs[index] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, divExpr,
          ArrayRef<OpFoldResult>{ivs[ivIndex++], tileSizes[reductionDim]});
    }
  }
  return splitReductionIvs;
}

//===----------------------------------------------------------------------===//
// tileToPartialReduction implementation.
//===----------------------------------------------------------------------===//

struct InitSliceInfo {
  SmallVector<int64_t> resultShape;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

/// In a given set vector, get the position of a particular element.
std::optional<int> getPositionIn(const llvm::SetVector<unsigned> &reductionDims,
                                 unsigned value) {
  for (auto [index, reductionDim] : llvm::enumerate(reductionDims)) {
    if (reductionDim == value) {
      return index;
    }
  }
  return std::nullopt;
}

/// Return the result shape, offsets, sizes and strides of the slice of the
/// `initValue` to use as the destination of the partial reduction op generated
/// with outer reduction strategy.
static InitSliceInfo getInitSliceInfoForOuterReduction(
    MLIRContext *context, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, const SetVector<unsigned> &reductionDims,
    ArrayRef<OpFoldResult> splitReductionIvs, AffineMap partialReductionMap) {
  int64_t initRank = partialReductionMap.getNumResults();
  SmallVector<OpFoldResult> initOffsets, initSizes;
  Attribute zero = IntegerAttr::get(IndexType::get(context), 0);
  Attribute one = IntegerAttr::get(IndexType::get(context), 1);
  SmallVector<OpFoldResult> initStrides(initRank, one);
  for (AffineExpr dimExpr : partialReductionMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(dimExpr).getPosition();
    if (reductionDims.contains(dim)) {
      initOffsets.push_back(zero);
    } else {
      initOffsets.push_back(offsets[dim]);
    }
    initSizes.push_back(sizes[dim]);
  }
  SmallVector<int64_t> resultShape;
  std::tie(resultShape, std::ignore) = decomposeMixedValues(initSizes);
  return {resultShape, initOffsets, initSizes, initStrides};
}

/// Return the result shape, offsets, sizes and strides of the slice of the
/// `initValue` to use as destination of the partial reduction op generated with
/// outer parallel strategy.
static InitSliceInfo getInitSliceInfoForOuterParallel(
    MLIRContext *context, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, const SetVector<unsigned> &reductionDims,
    ArrayRef<OpFoldResult> splitReductionIvs, AffineMap partialReductionMap) {
  int64_t initRank = partialReductionMap.getNumResults();
  SmallVector<OpFoldResult> initOffsets, initSizes;
  Attribute one = IntegerAttr::get(IndexType::get(context), 1);
  SmallVector<OpFoldResult> initStrides(initRank, one);
  SmallVector<OpFoldResult> resultShape;
  for (AffineExpr dimExpr : partialReductionMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(dimExpr).getPosition();
    if (std::optional<unsigned> dimPos = getPositionIn(reductionDims, dim)) {
      initOffsets.push_back(splitReductionIvs[dimPos.value()]);
      initSizes.push_back(one);
    } else {
      initOffsets.push_back(offsets[dim]);
      initSizes.push_back(sizes[dim]);
      resultShape.push_back(sizes[dim]);
    }
  }
  SmallVector<int64_t> staticShapes;
  std::tie(staticShapes, std::ignore) = decomposeMixedValues(resultShape);
  return {staticShapes, initOffsets, initSizes, initStrides};
}

/// Return the result shape, offsets, sizes and strides of the slice of the
/// `initValue` to use as destination of the partial reduction op.
static InitSliceInfo getInitSliceInfo(MLIRContext *context,
                                      ReductionTilingStrategy strategy,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes,
                                      const SetVector<unsigned> &reductionDims,
                                      ArrayRef<OpFoldResult> splitReductionIvs,
                                      AffineMap partialReductionMap) {
  if (strategy == ReductionTilingStrategy::PartialReductionOuterReduction) {
    return getInitSliceInfoForOuterReduction(context, offsets, sizes,
                                             reductionDims, splitReductionIvs,
                                             partialReductionMap);
  }
  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
         "unexpected ReductionTilingStrategy");
  return getInitSliceInfoForOuterParallel(context, offsets, sizes,
                                          reductionDims, splitReductionIvs,
                                          partialReductionMap);
}

/// Return an AffineMaps to use for the `outs` operands of the linalg op
/// generated for partial results. The new AffineMap is the AffineMap of the
/// untiled op with reduction dimensions appended at end in order in which they
/// were specified during tiling.
static SmallVector<AffineMap>
getPartialResultAffineMaps(LinalgOp linalgOp,
                           const SetVector<unsigned> &reductionDims) {
  auto partialReductionMaps = llvm::map_to_vector(
      linalgOp.getDpsInitsMutable(), [&](OpOperand &opOperand) {
        AffineMap map = linalgOp.getMatchingIndexingMap(&opOperand);
        for (auto redPos : reductionDims) {
          map =
              map.insertResult(getAffineDimExpr(redPos, linalgOp.getContext()),
                               map.getNumResults());
        }
        return map;
      });
  return partialReductionMaps;
}

FailureOr<TilingResult>
tileToPartialReduction(Operation *op, OpBuilder &b, Location loc,
                       ReductionTilingStrategy tilingStrategy, ValueRange init,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       const SetVector<unsigned> &reductionDims,
                       ArrayRef<OpFoldResult> splitReductionIvs) {

  OpBuilder::InsertionGuard guard(b);
  auto linalgOp = cast<LinalgOp>(op);

  // Canditate for moving init operands (outs) to ins.
  auto outputOperands = linalgOp.getDpsInitsMutable();
  SetVector<OpOperand *> candidates;
  for (OpOperand &op : outputOperands) {
    if (linalgOp.getMatchingBlockArgument(&op).use_empty()) {
      continue;
    }
    candidates.insert(&op);
  }

  SmallVector<AffineMap> partialReductionMaps =
      getPartialResultAffineMaps(linalgOp, reductionDims);

  // Step 1. Extend init maps to have reduction dimension dims, since we
  // are converting them to parallel dimensions.
  SmallVector<AffineMap> newInitMaps;
  if (tilingStrategy ==
      ReductionTilingStrategy::PartialReductionOuterReduction) {
    newInitMaps = llvm::to_vector(partialReductionMaps);
  } else {
    newInitMaps = llvm::map_to_vector(
        linalgOp.getDpsInitsMutable(), [&](OpOperand &opOperand) {
          return linalgOp.getMatchingIndexingMap(&opOperand);
        });
  }

  // Step 2a: Extract a slice of the input operands.
  SmallVector<Value> tiledInputs = makeTiledShapes(
      b, loc, linalgOp, linalgOp.getDpsInputs(), offsets, sizes, {}, true);
  SmallVector<Operation *> generatedSlices = llvm::map_to_vector(
      llvm::make_filter_range(
          tiledInputs, [](Value v) -> bool { return v.getDefiningOp(); }),
      [](Value v) -> Operation * { return v.getDefiningOp(); });

  // Step 2b: Extract a slice of the init operands.
  SmallVector<Value, 1> tiledInits;
  for (auto [partialReductionMap, valueToTile] :
       llvm::zip_equal(partialReductionMaps, init)) {
    InitSliceInfo sliceInfo =
        getInitSliceInfo(b.getContext(), tilingStrategy, offsets, sizes,
                         reductionDims, splitReductionIvs, partialReductionMap);
    auto valueToTileType = cast<RankedTensorType>(valueToTile.getType());
    RankedTensorType sliceResultType = RankedTensorType::get(
        sliceInfo.resultShape, valueToTileType.getElementType(),
        valueToTileType.getEncoding());
    auto sliceOp = tensor::ExtractSliceOp::create(
        b, loc, sliceResultType, valueToTile, sliceInfo.offsets,
        sliceInfo.sizes, sliceInfo.strides);
    tiledInits.push_back(sliceOp.getResult());
    generatedSlices.push_back(sliceOp);
  }

  // Update the indexing maps.
  SmallVector<AffineMap> newMaps;
  if (tilingStrategy ==
      ReductionTilingStrategy::PartialReductionOuterReduction) {
    // Add the tiledInits to inputs
    tiledInputs.append(tiledInits.begin(), tiledInits.end());

    // Update the indexing maps: Inputs -- Init operand moved to ins -- outs
    // Init operands are usually stored as part of outs for reduction loop
    int64_t origNumInput = linalgOp.getNumDpsInputs();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    newMaps.append(indexingMaps.begin(),
                   std::next(indexingMaps.begin(), origNumInput));

    for (auto [initOperand, newInitMap] :
         llvm::zip_equal(linalgOp.getDpsInitsMutable(), newInitMaps)) {
      if (candidates.contains(&initOperand)) {
        newMaps.push_back(newInitMap);
      }
    }
    newMaps.append(newInitMaps.begin(), newInitMaps.end());
  } else {
    newMaps = linalgOp.getIndexingMapsArray();
    for (auto [initOperand, newInitMap] :
         llvm::zip_equal(linalgOp.getDpsInitsMutable(), newInitMaps)) {
      int mapIdx = linalgOp.getIndexingMapIndex(&initOperand);
      newMaps[mapIdx] = newInitMap;
    }
  }

  // Step 3. Change the reduction dim iterator types.
  SmallVector<utils::IteratorType> newIteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (tilingStrategy ==
      ReductionTilingStrategy::PartialReductionOuterReduction) {
    for (int dim : reductionDims)
      newIteratorTypes[dim] = utils::IteratorType::parallel;
  }

  // Step 4. Create the new generic op.
  Operation *partialReductionOp;
  auto resultTypes = ValueRange(tiledInits).getTypes();
  if (tilingStrategy ==
      ReductionTilingStrategy::PartialReductionOuterReduction) {
    auto newOp = b.create<GenericOp>(loc, resultTypes, tiledInputs, tiledInits,
                                     newMaps, newIteratorTypes);
    // Update the basic block after moving inits to ins operand.
    IRMapping mapping;
    Region &region = newOp.getRegion();
    Block *block = b.createBlock(&region);
    b.setInsertionPointToStart(block);

    for (auto bbarg : linalgOp.getRegionInputArgs()) {
      mapping.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }
    for (OpOperand *op : candidates) {
      BlockArgument bbarg = linalgOp.getMatchingBlockArgument(op);
      mapping.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }
    for (OpOperand &op : outputOperands) {
      BlockArgument bbarg = linalgOp.getMatchingBlockArgument(&op);
      if (candidates.count(&op))
        block->addArgument(bbarg.getType(), loc);
      else
        mapping.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }
    for (auto &op : linalgOp->getRegion(0).front().getOperations()) {
      b.clone(op, mapping);
    }
    partialReductionOp = newOp.getOperation();
  } else {
    SmallVector<Value> operands = std::move(tiledInputs);
    llvm::append_range(operands, tiledInits);
    partialReductionOp = mlir::clone(b, op, resultTypes, operands);
  }
  return TilingResult{
      {partialReductionOp},
      llvm::map_to_vector(partialReductionOp->getResults(),
                          [](OpResult r) -> Value { return r; }),
      generatedSlices};
}

//===----------------------------------------------------------------------===//
// End - tileToPartialReduction implementation.
//===----------------------------------------------------------------------===//

static FailureOr<TilingResult>
getTiledImplementation(RewriterBase &rewriter, TilingInterface op,
                       ReductionTilingStrategy reductionStrategy,
                       ValueRange regionIterArg, ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes, ValueRange ivs,
                       ArrayRef<OpFoldResult> numThreads,
                       ArrayRef<OpFoldResult> tileSizes,
                       const SetVector<unsigned> &reductionDims) {
  if (reductionStrategy == ReductionTilingStrategy::FullReduction) {
    return op.getTiledImplementation(rewriter, offsets, sizes);
  }

  auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
  if (!redOp) {
    return rewriter.notifyMatchFailure(
        op, "PartialReductionOuterReduction tiling strategy is only "
            "supported for operations "
            "implementing PartialReductionOpInterface");
  }

  SmallVector<OpFoldResult> splitReductionIvs =
      getSplitReductionIvs(rewriter, op.getLoc(), reductionStrategy, ivs,
                           numThreads, tileSizes, reductionDims);
  return tileToPartialReduction(redOp, rewriter, op.getLoc(), reductionStrategy,
                                regionIterArg, offsets, sizes, reductionDims,
                                splitReductionIvs);
}

static LogicalResult getResultTilePosition(
    RewriterBase &rewriter, ReductionTilingStrategy reductionStrategy,
    int64_t index, Value tiledResult, TilingInterface op,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ValueRange ivs, ArrayRef<OpFoldResult> numThreads,
    ArrayRef<OpFoldResult> tileSizes, const SetVector<unsigned> &reductionDims,
    SmallVector<OpFoldResult> &resultOffset,
    SmallVector<OpFoldResult> &resultSize) {

  if (reductionStrategy == ReductionTilingStrategy::FullReduction) {
    return op.getResultTilePosition(rewriter, index, offsets, sizes,
                                    resultOffset, resultSize);
  }
  auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
  if (!redOp) {
    return rewriter.notifyMatchFailure(
        op, "PartialReductionOuterReduction tiling strategy is only supported"
            "for operations implementing PartialReductionOpInterface");
  }
  SmallVector<OpFoldResult> splitReductionIvs =
      getSplitReductionIvs(rewriter, op.getLoc(), reductionStrategy, ivs,
                           numThreads, tileSizes, reductionDims);
  return redOp.getPartialResultTilePosition(
      rewriter, index, reductionStrategy, offsets, sizes, reductionDims,
      splitReductionIvs, resultOffset, resultSize);
}

static FailureOr<MergeResult>
mergeTilingResults(RewriterBase &rewriter, TilingInterface op,
                   ReductionTilingStrategy reductionStrategy,
                   const SetVector<unsigned> &reductionDims,
                   ValueRange partialResults) {
  assert(reductionStrategy != ReductionTilingStrategy::FullReduction &&
         "expected merge to be called for only partial reduction cases");

  auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
  if (!redOp) {
    return rewriter.notifyMatchFailure(
        op, "PartialReductionOuterReduction tiling strategy is only "
            "supported for operations "
            "implementing PartialReductionOpInterface");
  }
  return redOp.mergeReductions(rewriter, op.getLoc(), partialResults,
                               reductionDims);
}

/// Append the specified additional `newInitOperands` operands to the
/// loops existing `init` operands (or similar), and replace `loopOp` with
/// the new loop that has the additional init operands. The loop body of
/// this loop is moved over to the new loop. `yieldTiledValuesFn`
/// is called to get the new tiled values returned, and the offset
/// and sizes at which the tiled value is inserted into the
/// new region iter_args that correspond to the newly added init operands.
template <typename LoopType>
FailureOr<LoopLikeOpInterface>
yieldTiledValuesAndReplaceLoop(LoopType loopOp, RewriterBase &rewriter,
                               ValueRange newInitOperands,
                               YieldTiledValuesFn yieldTiledValuesFn) {
  return rewriter.notifyMatchFailure(loopOp, "unhandled loop type");
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for `scf.for`.
template <>
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop<scf::ForOp>(
    scf::ForOp loopOp, RewriterBase &rewriter, ValueRange newInitOperands,
    YieldTiledValuesFn yieldTiledValuesFn) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = loopOp.getLoc();
  rewriter.setInsertionPoint(loopOp);

  auto inits = llvm::to_vector(loopOp.getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  auto newLoop = scf::ForOp::create(
      rewriter, loc, loopOp.getLowerBound(), loopOp.getUpperBound(),
      loopOp.getStep(), inits, [](OpBuilder &, Location, Value, ValueRange) {},
      loopOp.getUnsignedCmp());

  // Move the loop body to the new op.
  Block *loopBody = loopOp.getBody();
  Block *newLoopBody = newLoop.getBody();
  rewriter.mergeBlocks(
      loopBody, newLoopBody,
      newLoopBody->getArguments().take_front(loopBody->getNumArguments()));

  auto yieldOp = cast<scf::YieldOp>(newLoopBody->getTerminator());
  rewriter.setInsertionPoint(yieldOp);

  SmallVector<Value> tiledValues;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  ValueRange newRegionIterArgs =
      newLoop.getRegionIterArgs().take_back(newInitOperands.size());
  if (failed(yieldTiledValuesFn(rewriter, loc, newLoop.getInductionVar(),
                                newRegionIterArgs, tiledValues, resultOffsets,
                                resultSizes))) {
    rewriter.eraseOp(newLoop);
    return rewriter.notifyMatchFailure(loopOp, "failed to get tiled values");
  }

  SmallVector<Value> newYieldValues = llvm::to_vector(yieldOp.getOperands());
  for (auto [tiledValue, regionIterArg, resultOffset, resultSize] :
       llvm::zip_equal(tiledValues, newRegionIterArgs, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    Value insert = tensor::InsertSliceOp::create(
        rewriter, yieldOp->getLoc(), tiledValue, regionIterArg, resultOffset,
        resultSize, resultStride);
    newYieldValues.push_back(insert);
  }

  rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldValues);
  rewriter.replaceOp(loopOp,
                     newLoop->getResults().take_front(loopOp.getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for `scf.forall`
template <>
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop<scf::ForallOp>(
    scf::ForallOp loopOp, RewriterBase &rewriter, ValueRange newInitOperands,
    YieldTiledValuesFn yieldTiledValuesFn) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = loopOp.getLoc();
  rewriter.setInsertionPoint(loopOp);
  auto inits = llvm::to_vector(loopOp.getOutputs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  auto newLoop = scf::ForallOp::create(
      rewriter, loc, loopOp.getMixedLowerBound(), loopOp.getMixedUpperBound(),
      loopOp.getMixedStep(), inits, loopOp.getMapping(),
      [](OpBuilder &, Location, ValueRange) {});

  // Move the region of the current block to the newly created op.
  Block *loopBody = loopOp.getBody();
  Block *newLoopBody = newLoop.getBody();
  rewriter.mergeBlocks(
      loopBody, newLoopBody,
      newLoopBody->getArguments().take_front(loopBody->getNumArguments()));

  auto terminator = cast<scf::InParallelOp>(newLoopBody->getTerminator());
  rewriter.setInsertionPoint(terminator);
  SmallVector<Value> tiledValues;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  ValueRange regionIterArgs =
      newLoop.getRegionIterArgs().take_back(newInitOperands.size());
  if (failed(yieldTiledValuesFn(rewriter, loc, newLoop.getInductionVars(),
                                regionIterArgs, tiledValues, resultOffsets,
                                resultSizes))) {
    rewriter.eraseOp(newLoop);
    return rewriter.notifyMatchFailure(loopOp,
                                       "failed to get yielded tiled values");
  }

  // Update the terminator.
  rewriter.setInsertionPointToEnd(terminator.getBody());

  for (auto [tiledValue, iterArg, resultOffset, resultSize] : llvm::zip_equal(
           tiledValues, regionIterArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    tensor::ParallelInsertSliceOp::create(rewriter, terminator.getLoc(),
                                          tiledValue, iterArg, resultOffset,
                                          resultSize, resultStride);
  }

  rewriter.replaceOp(loopOp,
                     newLoop->getResults().take_front(loopOp.getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for
/// `LoopLikeOpInterface`, that just dispatches to the implementation for each
/// supported loop type.
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop(
    LoopLikeOpInterface loopLikeOp, RewriterBase &rewriter,
    ValueRange newInitOperands, YieldTiledValuesFn yieldTiledValuesFn) {
  return TypeSwitch<Operation *, FailureOr<LoopLikeOpInterface>>(
             loopLikeOp.getOperation())
      .Case<scf::ForOp, scf::ForallOp>(
          [&](auto loopOp) -> FailureOr<LoopLikeOpInterface> {
            return yieldTiledValuesAndReplaceLoop(
                loopOp, rewriter, newInitOperands, yieldTiledValuesFn);
          })
      .Default([&](auto loopOp) -> FailureOr<LoopLikeOpInterface> {
        return rewriter.notifyMatchFailure(loopOp, "unhandled loop type");
      });
}

/// Method to add new init values to a loop nest. Updates `loops` in-place
/// with new loops that use the `newInitValues`. The outer-loops are updated
/// to yield the new result values of the inner loop. For the innermost loop,
/// the call back `getNewYields` is invoked to get the additional values to
/// yield form the innermost loop.
static LogicalResult addInitOperandsToLoopNest(
    RewriterBase &rewriter, MutableArrayRef<LoopLikeOpInterface> loops,
    ValueRange newInitValues, YieldTiledValuesFn getNewTiledYieldsFn) {
  if (loops.empty())
    return success();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loops.front());

  SmallVector<Value> ivs;
  for (auto &loop : loops.drop_back()) {
    rewriter.setInsertionPoint(loop);

    // if loops.size() > 1 we assume that scf.for is used for the loops.
    auto forLoop = cast<scf::ForOp>(loop.getOperation());

    // Create a new loop with the new init values for this loop.
    SmallVector<Value> newInits = llvm::to_vector(forLoop.getInitArgs());
    newInits.append(newInitValues.begin(), newInitValues.end());
    auto newLoop = scf::ForOp::create(
        rewriter, forLoop.getLoc(), forLoop.getLowerBound(),
        forLoop.getUpperBound(), forLoop.getStep(), newInits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {},
        forLoop.getUnsignedCmp());

    // Merge the body of the new loop with the body of the old loops.
    SmallVector<Value> sourceBlockArgs;
    sourceBlockArgs.push_back(newLoop.getInductionVar());
    auto newRegionIterArgs = newLoop.getRegionIterArgs();
    sourceBlockArgs.append(
        newRegionIterArgs.begin(),
        std::next(newRegionIterArgs.begin(), forLoop.getNumResults()));
    rewriter.mergeBlocks(forLoop.getBody(), newLoop.getBody(), sourceBlockArgs);
    rewriter.replaceOp(
        forLoop, newLoop.getResults().take_front(forLoop.getNumResults()));
    loop = newLoop;
    ivs.push_back(newLoop.getInductionVar());
    newInitValues = newLoop.getRegionIterArgs().take_back(newInitValues.size());
  }

  // Update the loop body of the innermost loop to get new yield values.
  LoopLikeOpInterface innerMostLoop = loops.back();
  FailureOr<LoopLikeOpInterface> newInnerMostLoop =
      yieldTiledValuesAndReplaceLoop(innerMostLoop, rewriter, newInitValues,
                                     getNewTiledYieldsFn);

  if (failed(newInnerMostLoop))
    return innerMostLoop.emitOpError("failed to return additional yields");
  loops.back() = newInnerMostLoop.value();

  // Make all other loops except the innermost loops yield the values returned
  // by the inner loop.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(loops.drop_back(), loops.drop_front())) {
    // Again assume that all the outer loops are scf.for operations.
    auto outerForLoop = cast<scf::ForOp>(outerLoop);
    auto outerLoopYield =
        cast<scf::YieldOp>(outerForLoop.getBody()->getTerminator());
    SmallVector<Value> newYields =
        llvm::to_vector(outerLoopYield.getOperands());
    ValueRange additionalYields =
        innerLoop->getResults().take_back(newInitValues.size());
    newYields.append(additionalYields.begin(), additionalYields.end());
    rewriter.setInsertionPoint(outerLoopYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
  }
  return success();
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
FailureOr<scf::SCFTilingResult>
TileLoopUsingSCF(RewriterBase &rewriter, TilingInterface op,
                 const scf::SCFTilingOptions &options) {
  if (failed(verifyOptions(rewriter, op.getLoc(), options))) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);

  // 2. Materialize the tile sizes and/or number of threads;
  SmallVector<OpFoldResult> tileSizes, numThreads;
  std::tie(tileSizes, numThreads) =
      getUserTileSizesAndNumThreads(rewriter, op, iterationDomain, options);

  // Check if it is safe to tile. This is hold over from previous iterations
  // of tile to for-all. Consider dropping it.
  if (failed(checkTileSizes(op, options.loopType, options.reductionStrategy,
                            tileSizes, numThreads))) {
    return failure();
  }

  // Get the reduction dims
  SetVector<unsigned> reductionDims =
      getSanitizedReductionDims(tileSizes, options);

  // 3. If there is an interchange specified, permute the iteration domain and
  // the tile sizes.
  SmallVector<int64_t> interchangeVector;
  if (!options.interchangeVector.empty()) {
    interchangeVector = fillInterchangeVector(options.interchangeVector,
                                              iterationDomain.size());
    assert(isPermutationVector(interchangeVector) &&
           "expected interchange vector to be a permutation");

    applyPermutationToVector(iterationDomain, interchangeVector);
    applyPermutationToVector(tileSizes, interchangeVector);
    if (!numThreads.empty())
      applyPermutationToVector(numThreads, interchangeVector);
  }

  FailureOr<TilingResult> tilingResult;
  // 4. Define the lambda function used later to generate the body of the
  // innermost tiled loop.
  YieldTiledValuesFn innerYieldTiledValuesFn =
      [&](RewriterBase &rewriter, Location loc, ValueRange ivs,
          ValueRange regionIterArgs, SmallVector<Value> &tiledResults,
          SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
          SmallVector<SmallVector<OpFoldResult>> &resultSizes)
      -> LogicalResult {
    // 4a. Compute the `offsets` and `sizes` to use for tiling.
    SmallVector<OpFoldResult> offsets, sizes;
    std::tie(offsets, sizes) = getTileOffsetAndSizes(
        rewriter, loc, options.reductionStrategy, ivs, iterationDomain,
        tileSizes, numThreads, reductionDims);

    // 4b. If interchange was provided, apply inverse of the interchange
    //     to get back the offsets/sizes in the order to be specified.
    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      applyPermutationToVector(offsets, inversePermutation);
      applyPermutationToVector(sizes, inversePermutation);
    }

    // 5. Generate the tiled implementation within the inner most loop.

    // 5a. Clone the operation within the loop body.
    auto clonedOp = cast<TilingInterface>(
        cloneOpAndUpdateDestinationArgs(rewriter, op, regionIterArgs));

    // 5b. Early return cloned op if tiling is not happening. We can not
    // return the original op because it could lead to `rewriter.replaceOp(op,
    // op->getResults())` and users would get crash.
    if (llvm::all_of(tileSizes, isZeroInteger)) {
      tiledResults.append(clonedOp->result_begin(), clonedOp->result_end());
      tilingResult =
          TilingResult{/*tiledOps=*/{clonedOp}, clonedOp->getResults(),
                       /*generatedSlices=*/{}};
      return success();
    }

    // 5c. Tile the cloned operation.
    tilingResult = getTiledImplementation(
        rewriter, clonedOp, options.reductionStrategy, regionIterArgs, offsets,
        sizes, ivs, numThreads, tileSizes, reductionDims);
    if (failed(tilingResult)) {
      rewriter.eraseOp(clonedOp);
      return op.emitOpError("faild to tile operation");
    }

    // 5d. Delete the cloned operation.
    rewriter.eraseOp(clonedOp);

    // 5e. Compute the offsets at which the result values are to be inserted
    //     back into its destinations.
    for (auto [index, tiledValue] :
         llvm::enumerate(tilingResult->tiledValues)) {
      tiledResults.push_back(tiledValue);
      SmallVector<OpFoldResult> resultOffset, resultSize;
      if (failed(getResultTilePosition(
              rewriter, options.reductionStrategy, index, tiledValue, op,
              offsets, sizes, ivs, numThreads, tileSizes, reductionDims,
              resultOffset, resultSize))) {
        for (auto op : tilingResult->tiledOps) {
          rewriter.eraseOp(op);
        }
        return rewriter.notifyMatchFailure(
            op, "failed to get slice of result produced");
      }
      resultOffsets.emplace_back(std::move(resultOffset));
      resultSizes.emplace_back(std::move(resultSize));
    }

    return success();
  };

  // 6. Find the destination tensors to use for the operation.
  FailureOr<SmallVector<Value>> maybeInits = createInitialTensorsForTiling(
      rewriter, op, options.reductionStrategy, iterationDomain, numThreads,
      tileSizes, reductionDims);
  if (failed(maybeInits)) {
    return rewriter.notifyMatchFailure(
        op, "unable to create initial tensors for tiling");
  }
  SmallVector<Value> &initTensors = maybeInits.value();

  // 7. Generate the tiled loops nest using the callback defined above.
  SmallVector<LoopLikeOpInterface> loops;
  if (failed(generateLoopNest(rewriter, op.getLoc(), options.loopType,
                              iterationDomain, tileSizes, numThreads,
                              initTensors, options.mappingVector,
                              innerYieldTiledValuesFn, loops)))
    return op.emitOpError("failed to generate tiling loops");
  assert(succeeded(tilingResult) &&
         "expected tiling result to be computed after loop generation");

  if (loops.empty()) {
    // If loops are empty, the tiled op is used as the replacement for the
    // untiled op.
    return scf::SCFTilingResult{tilingResult->tiledOps,
                                initTensors,
                                loops,
                                tilingResult->tiledValues,
                                tilingResult->generatedSlices,
                                {}};
  }

  auto loopResults = llvm::map_to_vector(loops.front()->getResults(),
                                         [](OpResult r) -> Value { return r; });

  // For the full reduction case, there is nothing more to do.
  if (options.reductionStrategy == ReductionTilingStrategy::FullReduction) {
    return scf::SCFTilingResult{
        tilingResult->tiledOps,        initTensors, loops, loopResults,
        tilingResult->generatedSlices, {}};
  }

  // The results of the loop needs to be merged.
  FailureOr<MergeResult> mergeResult = mergeTilingResults(
      rewriter, op, options.reductionStrategy, reductionDims, loopResults);
  if (failed(mergeResult)) {
    return rewriter.notifyMatchFailure(
        op, "Failed to merge partial results from tiling");
  }
  return scf::SCFTilingResult{tilingResult->tiledOps,
                              initTensors,
                              loops,
                              mergeResult->replacements,
                              tilingResult->generatedSlices,
                              mergeResult->mergeOps};
}

FailureOr<scf::SCFTilingResult>
TileReductionLoop(RewriterBase &b, PartialReductionOpInterface op,
                  ArrayRef<OpFoldResult> tileSize) {
  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterReduction);
  options.setTileSizes(tileSize);
  SmallVector<unsigned> reductionDims;
  for (auto [index, iteratorType] : llvm::enumerate(op.getLoopIteratorTypes()))
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(index);
  options.setReductionDims(reductionDims);
  return TileLoopUsingSCF(b, op, options);
}
} // namespace

namespace mlir {
namespace hexagon {

// Split the reduction loop into parallel computation,
// followed by reduction on the final output.
LogicalResult SplitReductionLinalgOp(linalg::LinalgOp generalizeOp) {

  // Operation which have PartialReductionOpInterface
  // are candidates for split reduction
  auto partialReductionOp =
      cast<PartialReductionOpInterface>(generalizeOp.getOperation());

  // Early exit when there is no partialReductionOp
  if (!partialReductionOp) {
    DBG("-> Op doesn't support split reduction.\n");
    return failure();
  }

  IRRewriter rewriter(generalizeOp.getContext());
  rewriter.setInsertionPoint(generalizeOp.getOperation());
  unsigned numLoops = generalizeOp.getNumLoops();

  // Compute tile size to split the reduction loop
  SmallVector<int64_t, 10> tileSizes(numLoops, 0);
  auto dataTileSize = computeDataTileSize(generalizeOp);

  if (!dataTileSize.has_value())
    return failure();

  tileSizes[numLoops - 1] = dataTileSize.value();

  // Apply split reduction on the reduction loop
  FailureOr<scf::SCFTilingResult> result =
      TileReductionLoop(rewriter, partialReductionOp,
                        getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes)));

  if (failed(result)) {
    return failure();
  }

  // Replace with the new op
  rewriter.replaceOp(generalizeOp, result->replacements);
  return success();
}

bool IsSplitReductionCandidate(linalg::LinalgOp op) {
  auto generalizedOp = cast<linalg::GenericOp>(op);
  for (Operation &op : generalizedOp.getBody()->getOperations()) {
    if (!(isa<arith::AddFOp>(op) || isa<linalg::YieldOp>(op) ||
          isa<arith::MaxNumFOp>(op) || isa<arith::MinNumFOp>(op) ||
          isa<arith::MulFOp>(op))) {
      return false;
    }
  }
  return true;
}
} // namespace hexagon
} // namespace mlir
