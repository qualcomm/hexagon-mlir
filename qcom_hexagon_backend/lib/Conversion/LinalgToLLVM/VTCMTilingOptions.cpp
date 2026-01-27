//===- VTCMTilingOptions.cpp - tiling for vtcm options --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file adds helper functions for setting tiling options for vtcm tiling
// such as tile size estimation.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "vtcm-tiling-options"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

namespace {

// Given a list of affine-maps and an equal sized list of constants,
// returns sum of product of dim-results multiplied by constants.
// e.g. given [affine_map<(d0, d1) ->(d0, d1>, affine_map<(d0, d1) -> (d1)>]
// and constants c0, c1; returns
// affine_map<(d0,d1) -> d0*d1*c0 + d1*c1)
// Does not handle symbols currently.
AffineMap flattenAndfuseAffineMapsWithConstants(
    SmallVector<AffineMap> maps, SmallVector<int64_t> constantFactors,
    MLIRContext *context, const int64_t zeroExpressionValue = 0) {

  // Returns an empty AffineMap if there are no input maps
  if (maps.empty())
    return AffineMap::get(context);

  unsigned numResults = 1;
  unsigned numDims = maps.front().getNumDims();
  unsigned numSymbols = 0;

  assert(maps.size() == constantFactors.size() && "expect sizes to match");
  assert(
      llvm::all_of(maps, [](AffineMap m) { return m.getNumSymbols() == 0; }) &&
      "Affine maps with symbols is invalid.");

  AffineExpr result = getAffineConstantExpr(0, context);
  for (auto [mIdx, m] : llvm::enumerate(maps)) {
    if (m.isEmpty())
      continue;

    // For each non-empty map generate a subExpression which is the product of
    // the result expressions of the map, and a corresponding constant factor.
    // If any result expression is a zero value constant, then that is replaced
    // with the zeroExpressionValue constant in the computation.
    AffineExpr subExpr = getAffineConstantExpr(constantFactors[mIdx], context);
    for (const auto &[resIdx, expr] : llvm::enumerate(m.getResults())) {
      if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
        if (constExpr.getValue() == 0) {
          subExpr =
              subExpr * getAffineConstantExpr(zeroExpressionValue, context);
        } else {
          subExpr = subExpr * constExpr;
        }
      } else {
        subExpr = subExpr * expr;
      }
    }
    result = result + subExpr;
  }
  return AffineMap::get(numDims, numSymbols, {result}, context);
}

// Given a Linalg Op, to generate a mapping from the loop bounds to the memory
// footprint of the operation. The resulting affine map is the symbolic
// representation of the sum of the memory footprint of each operand
// in the linalg op. For example:
//
// ```
// #map = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d1)>
// linalg.generic {indexing_maps = [#map1, #map],
//     iterator_types = ["reduction", "parallel"]}
//     ins(%x : tensor<256x128xf32>)
//     outs(%y : tensor<128xf32>) { ... }
//```
//
// Memory footprint affine map: affine_map<(d0, d1) -> (d1 * 4 + d0 * d1 * 4)>
std::optional<AffineMap> generateMemoryFootprintAffineMap(linalg::LinalgOp op) {

  auto maps = op.getIndexingMapsArray();
  SmallVector<int64_t> data_sizes;
  for (auto t : op->getOperandTypes()) {
    if (auto operandElementSize = getElementSizeInBytes(t)) {
      data_sizes.push_back(*operandElementSize);
    } else {
      DBG("Invalid operand type, can't calculate size");
      return std::nullopt;
    }
  }

  // If an operation has zero in the result of an affine map, then that
  // indicates broadcasting dims. Overriding the zero expression value to 1
  // means that we are assuming that the broadcasted dims are of unit size. So
  // for now, (d0,d1)->(d0,0) has the same memory footprint as (d0,d1)->(d0).
  return flattenAndfuseAffineMapsWithConstants(maps, data_sizes,
                                               op.getContext(), 1);
}

// Given the effective loop bounds and memory footprint affine map, applies the
// loop bounds over the affine map to get the memory footprint size of the
// operation.
//
// Given the following loop bounds and affine map we get,
// Loop bounds: [1024,256]
// Affine Map: (d0,d1)->(4* d0 * d1)
// Memory footprint: 4 * 1024 * 256 = 1048576
int64_t calcMemoryFootprint(linalg::LinalgOp op,
                            SmallVector<int64_t> loopBounds,
                            AffineMap memoryFootprintMap) {
  assert(loopBounds.size() == op.getStaticLoopRanges().size() &&
         "Loop bounds size mismatch with the operation.");
  assert(memoryFootprintMap.getNumResults() == 1 &&
         "Invalid memory footprint map");
  return memoryFootprintMap.compose(loopBounds).front();
};

// Sets the operands which are full tensors after applying the given tile sizes,
// i.e., the shape of the operand is the same before and after tiling.
void setFullTensors(linalg::LinalgOp op, SmallVector<int64_t> tileSizes,
                    SmallVector<bool> &fullTensorOperands) {
  assert(tileSizes.size() == op.getStaticLoopRanges().size() &&
         "Loop bounds size mismatch with the operation.");

  for (OpOperand &opOperand : op->getOpOperands()) {
    auto shape = op.getShape(&opOperand);
    auto indexMap = op.getMatchingIndexingMap(&opOperand);
    auto tiledShape = indexMap.compose(tileSizes);

    // In some cases the operand affine map has zero result since it is
    // independent of dimensions. This results in tiled shape having zero dim
    // whereas the actual shape after tiling would have unit dim.
    for (int i = 0; i < tiledShape.size(); ++i)
      if (!tiledShape[i])
        tiledShape[i] = 1;

    if (llvm::equal(shape, tiledShape)) {
      fullTensorOperands[opOperand.getOperandNumber()] = true;
    }
  }
  DBG("-> Full tensor operands: {" << toString(fullTensorOperands) << "}");
};

// Calculates the reduction factor as the ratio of the input size in bytes to
// vctm size (if input size is greater than vtcm size).
//
// TODO: We are updating the reduction factor as a power of 2, so that tile size
// generation is easier when tensor sizes are also of power of 2. Can explore
// if this constraint can be relaxed to use the ratio directly.
int64_t getReductionFactor(int64_t size) {

  if (size < vtcmSizeInBytes) {
    DBG("Input size is smaller than than vtcm size. Hence, setting reduction "
        "factor to 1.");
    return 1;
  }
  auto ratio = static_cast<double>(size) / vtcmSizeInBytes;

  // Even when the ratio is marginally greater than power of 2,
  // it would be updated to the next power of 2. Due to this, tile sizes are
  // generated which may be more conservative than required.
  unsigned reductionFactor =
      1 << static_cast<int64_t>(std::ceil(std::log2(ratio)));
  return reductionFactor;
}

// Given a Linalg Op and the corresponding memory footprint affine map,
// generates the priority ordering for the loop dimensions to be used during
// tiling. Using the original loop bounds for the op, reduces the size of each
// dim by a common factor one at a time, and calculates the memory footprint.
// This would determine the effectiveness of tiling that dim. These results are
// sorted seperately for set of reduction and parallel dims and then added to an
// ordering list in the order of smallest to largest memory footprint.
//
// TODO: The priority ordering is split for parallel and reduction dims, because
// it is assumed that tiling reduction dims would result in sub-optimal codegen.
// This can be fine-tuned for certain ops to determine the trade-off between
// performance reduction and impact of tiling the reduction dims. Then all the
// dims can be compared together for a more effective priority order.
//
// Use priorityDims to tile only a few user provided dimensions
SmallVector<int64_t>
generateTilingPriorityOrder(linalg::LinalgOp op,
                            AffineMap memoryFootprintAffineMap,
                            SmallVector<int64_t> priorityDims) {
  // If tiling dims are provided
  if (priorityDims.size()) {
    return priorityDims;
  }

  SmallVector<int64_t> ordering;
  SmallVector<int64_t> loopBounds = getInitialTileSize(op);
  for (int i = 0; i < 2; ++i) {
    SmallVector<unsigned int> dims;
    if (!i)
      op.getParallelDims(dims);
    else
      op.getReductionDims(dims);

    // Using a common reduction factor of 2 for each dimension to estimate the
    // memory footprint reduction.
    const int commonReductionFactor = 2;
    SmallVector<std::pair<int64_t, unsigned int>> priorityOrder;
    for (auto d : dims) {
      // Reducing the loop bounds only for the dimension in consideration
      // by the common factor, and then reverting after determining the
      // effective footprint.
      auto prevTileSize = loopBounds[d];
      loopBounds[d] /= commonReductionFactor;

      auto size = calcMemoryFootprint(op, loopBounds, memoryFootprintAffineMap);
      priorityOrder.push_back({size, d});

      loopBounds[d] = prevTileSize;
    }
    llvm::sort(priorityOrder);

    for (auto ele : priorityOrder) {
      ordering.push_back(ele.second);
    }
  }
  DBG("-> Priority dims ordering: {" << toString(ordering) << "}");
  return ordering;
}

// Check if a linalg op has inner tiles matching specific optimized shapes.
// If so, return the inner tile to be used as lower bounds for tiling.
std::optional<SmallVector<unsigned>>
getOptimizedTileLowerBounds(linalg::LinalgOp op,
                            SmallVector<int64_t> initalLoopRanges) {

  // Get element type from the first result
  auto resultTypes = op->getResultTypes();
  if (resultTypes.empty())
    return std::nullopt;
  auto firstType = resultTypes.front();
  auto elementType = getElementType(firstType);

  // Get optimized tile shape for the type if valid
  llvm::SmallVector<unsigned> tileShape;
  auto numLoops = initalLoopRanges.size();
  if (elementType.isInteger(8) && numLoops == 7)
    tileShape = INT8_TILE_SHAPE;
  else if (elementType.isF16() && numLoops == 8)
    tileShape = F16_TILE_SHAPE;
  else
    return std::nullopt;

  // To verify that the inner tiles of the linalg op fit the optimized tile
  // shape Note: Here we assume that operands have the same type and that there
  // is 1-1 affine-mapping for the iteration and data space for the inner tiles.
  // This is done because the checking should have been done by previous passes
  // such as LoweringToPack/SeedLayoutConversions, and is out of context for
  // tiling.
  auto innerTileRange = llvm::make_range(
      initalLoopRanges.end() - tileShape.size(), initalLoopRanges.end());
  if (llvm::equal(tileShape, innerTileRange))
    return tileShape;
  else
    return std::nullopt;
}

// Calculates the lower bounds to be used while determining tile size.
SmallVector<unsigned> calculateTileSizeLowerBounds(linalg::LinalgOp op) {

  SmallVector<unsigned> lowerBounds(op.getNumLoops(), 1);
  auto ranges = getInitialTileSize(op);

  // Parallelization factor corresponds to the parallelization of the
  // outermost loop across multiple threads/cores.
  // TODO: The parallelization factor is currently hardcoded but can be
  // determined for each op or passed as a parameter later on.
  unsigned parallelizationFactor = 4;
  if (ranges[0] >= parallelizationFactor)
    lowerBounds[0] = parallelizationFactor;

  if (auto optimizedLowerBounds = getOptimizedTileLowerBounds(op, ranges)) {
    assert(lowerBounds.size() >= optimizedLowerBounds->size() &&
           "Invalid optimized tile lower bounds");
    llvm::copy(*optimizedLowerBounds,
               lowerBounds.end() - optimizedLowerBounds->size());
  } else {
    // Using the vectorization size of the op (if valid) so that it can be
    // used as a lower bound for the tile size of the innermost loop.
    auto vectorizationSize = hexagon::computeDataTileSize(op);
    if (vectorizationSize.has_value()) {
      auto innermostDim = op.getNumLoops() - 1;
      if (ranges[innermostDim] >= vectorizationSize)
        lowerBounds[innermostDim] = vectorizationSize.value();
    }
  }
  DBG("-> Lower bounds: {" << toString(lowerBounds) << "}");
  return lowerBounds;
}
} // namespace

namespace mlir {
namespace hexagon {

// To calculate the tile size for a linalg op such that it fits in vtcm.
//
// Example:
// ```
// #map = affine_map<(d0, d1, d2) -> (d0, d1)>
// #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
// %.. = linalg.matmul ins(%arg0, %arg1 :
// tensor<2048x8192xf16>, tensor<8192x1024xf16>) outs(%arg2 :
// tensor<2048x1024xf32>)
// ```
//
// Calculate the memory footprint of the op. If memory footprint of op is
// greater than vtcm, determine reduction factor wrt vtcm size. Else, return
// original loop bounds as tile size.
//
// Memory footprint: 56MB > 2MB(vtcm size)
// Reduction factor = nextPowerOf2(56MB/2MB) = 32.
//
// Determine a priority ordering for the dims of the op, in the increasing
// order of effectiveness to tile that dim. To generate the ordering, each dim
// is tiled by a factor of 2 to determine the memory footprint. This is done
// seperately for parallel and reduction dims, and then merged to generate the
// final order.
//
// Parallel dims = [ 0, 2 ], Reduction dims = [ 1 ]
// Tiling dim 0 by 2, Tile size = [ 1024, 8192, 1024 ]
// Memory footprint (Tiling dim 0) = 36MB (1.56x reduction)
// Tiling dim 2 by 2, Tile size = [ 2048, 8192, 512 ]
// Memory footprint (Tiling dim 2) = 44MB (1.28x reduction)
// Tiling priority order with parallel dims = [ 0, 2 ]
// Tiling dim 1 by 2, Tile size = [ 2048 , 4096, 1024 ]
// Memory footprint (Tiling dim 1) = 32 MB (1.75x reduction)
// Final tiling priority dims order = [ 0,2,1 ]
//
// In priority order of dims, starting with the original loop bounds as the tile
// size, reduce the tile size at each dim (wrt constraints) by the reduction
// factor. Check if the op, when tiled with the tile size will fit in vtcm. If
// so, return the tile size. Else, generate a new reduction factor and repeat
// until memory footprint is less than vtcm size.
//
// Starting Tile size: [2048,8192,1024], RF:32
// Step 1: Tile Size: [64,8192,1024], Memory footprint: 17.25 > 2 -> RF=16
// Step 2: Tile Size: [64,8192,64], Memory footprint: 2.02 > 2 -> RF=2
// Step 3: Tile Size: [64,4096,64], Memory footprint: 1.02 < 2.
// Final Tile size: [64,4096,64]
//
// Next steps:
// - Add support for dynamic tensor shapes.
// - Optimize the priority order with layout constraints.
// - Along with generating the priority order, also generate the exact
// reduction factor for each dim. This would improve the efficiency of the tile
// size generated.
// - Add support for additional tile size constraints in addition to lower
// bounds.
std::optional<SmallVector<int64_t>>
determineTileSizes(linalg::LinalgOp op, SmallVector<int64_t> tilingDims) {

  auto memoryFootprintMap = generateMemoryFootprintAffineMap(op);
  if (!memoryFootprintMap)
    return std::nullopt;
  else {
    DBG("-> Memory footprint affine map for op is:" << *memoryFootprintMap);
  }

  auto size =
      calcMemoryFootprint(op, getInitialTileSize(op), *memoryFootprintMap);

  if (size <= vtcmSizeInBytes) {
    DBG("-> Memory footprint of op:"
        << size << " bytes, is less than or equal to VTCM Memory size: "
        << vtcmSizeInBytes
        << " bytes. Hence, tile size estimation for vtcm is not required.");
    return getInitialTileSize(op);
  } else {
    DBG("-> Memory footprint of op:"
        << size
        << " bytes, is greater than VTCM Memory size: " << vtcmSizeInBytes
        << " bytes. Starting, tile size estimation for vtcm.");
  }

  auto priorityDims =
      generateTilingPriorityOrder(op, *memoryFootprintMap, tilingDims);
  SmallVector<unsigned> lowerBounds = calculateTileSizeLowerBounds(op);

  // For each dim in priority list, the tile size is reduced by a reduction
  // factor with the minimum as lower bound constraint in the first pass. If
  // vtcm constraint is not satisfied, then lower bound constraint is removed.
  // TODO: The constraints and two-pass approach can be folded into the
  // priority ordering, which would mean that there are duplicates for dims in
  // the ordering list, but with different constraints or scenarios.
  SmallVector<int64_t> tileSizes = getInitialTileSize(op);
  int64_t reductionFactor = getReductionFactor(size);
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < priorityDims.size(); ++i) {
      DBG("-> Current tile size: {" << toString(tileSizes) << "}");
      DBG("-> Current reduction Factor is:" << reductionFactor);
      auto idx = priorityDims[i];
      unsigned lowerBound = !j ? lowerBounds[idx] : 1;
      unsigned ratio = tileSizes[idx] / reductionFactor;
      tileSizes[idx] = std::max(ratio, lowerBound);
      auto updatedSize =
          calcMemoryFootprint(op, tileSizes, *memoryFootprintMap);
      if (updatedSize <= vtcmSizeInBytes) {
        return tileSizes;
      } else {
        reductionFactor = getReductionFactor(updatedSize);
      }
    }
  }

  // even with unit size, the op does not fit in VTCM.
  return std::nullopt;
}

std::vector<unsigned> getInterchangeVector(linalg::LinalgOp op) {
  std::vector<unsigned> interchangeVector(op.getNumLoops(), 0);
  std::iota(interchangeVector.begin(), interchangeVector.end(), 0);
  return interchangeVector;
}

// Returns initial tile sizes for the linalgOp, wherein static-shape
// dimension is selected as initial size, and a default value for
// dynamic loop range dimension.
SmallVector<int64_t> getInitialTileSize(linalg::LinalgOp op) {
  const int dynamicShapeDimTileSize = 256;
  SmallVector<int64_t> initial_tile_size = op.getStaticLoopRanges();
  for (auto [idx, dim] : llvm::enumerate(initial_tile_size)) {
    if (ShapedType::isDynamic(dim))
      initial_tile_size[idx] = dynamicShapeDimTileSize;
  }
  return initial_tile_size;
}

std::optional<SmallVector<int64_t>>
getTileSizes(linalg::LinalgOp op,
             std::optional<SmallVector<int64_t>> userProvidedTileSizes) {
  SmallVector<int64_t> tileSizes;
  auto numLoops = op.getNumLoops();
  if (userProvidedTileSizes) {
    assert(userProvidedTileSizes.value().size() >= numLoops &&
           "User provided tile size has fewer dims than the op");
    SmallVector<int64_t> userProvidedTileSizesForOp(
        userProvidedTileSizes.value().end() - numLoops,
        userProvidedTileSizes.value().end());
    tileSizes = userProvidedTileSizesForOp;
  } else {
    auto tileSizesForOp = determineTileSizes(op);
    if (!tileSizesForOp) {
      DBG("-> Cannot determine tile size for op.");
      return std::nullopt;
    }
    tileSizes = *tileSizesForOp;
  }
  DBG("-> Tile size: {" << toString(tileSizes) << "}");
  return tileSizes;
};

FailureOr<linalg::LinalgTilingOptions>
getVTCMTilingOptions(linalg::LinalgOp op,
                     std::optional<SmallVector<int64_t>> userProvidedTileSizes,
                     SmallVector<bool> &prefetch) {

  DBG("getting vtcm tiling options for: " << op);
  if (!op.hasPureTensorSemantics() || !op->getNumResults() ||
      op.getNumLoops() < 1) {
    DBG("-> get vtcm tiling options aborted. Constraints not "
        "satisfied.");
    return failure();
  }

  linalg::LinalgTilingOptions options;
  options.setInterchange(getInterchangeVector(op));
  auto tileSizes = getTileSizes(op, userProvidedTileSizes);
  if (!tileSizes)
    return failure();
  setFullTensors(op, *tileSizes, prefetch);
  options.setTileSizes(*tileSizes);
  return options;
}
} // namespace hexagon
} // namespace mlir
