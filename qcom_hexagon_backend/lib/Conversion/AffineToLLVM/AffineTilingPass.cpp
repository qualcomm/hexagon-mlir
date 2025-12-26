//===----------- AffineTilingPass.cpp - Affine Tiling Pass ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements tiling of affine loops, using a heuristic to select
// tile sizes automatically based on cache size. The current heuristic just
// tries to fit the inner loop into the cache, dividing equally across
// dimensions (hypercube tiles).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/AffineToLLVM/Passes.h"

#define DEBUG_TYPE "affine-tiling"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using mlir::func::FuncOp;
using namespace mlir::affine;

#define GEN_PASS_DEF_AFFINETILING
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

namespace {

struct AffineTilingPass : public ::impl::AffineTilingBase<AffineTilingPass> {
public:
  explicit AffineTilingPass(const AffineTilingOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override;
  void getTileSizes(ArrayRef<AffineForOp> band,
                    SmallVectorImpl<unsigned> *tileSizes);

  constexpr static unsigned defaultTileSize = 32;
  constexpr static bool avoidMaxMinBounds = true;
};

/// Reduces each tile size to the largest divisor of the corresponding trip
/// count (if the trip count is known).
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() &&
         "wrong number of tile sizes for band");
  for (unsigned i = 0; i < band.size(); i++) {
    unsigned &tileSize = (*tileSizes)[i];
    if (std::optional<uint64_t> mayConst = getConstantTripCount(band[i])) {
      // Adjust the tile size to largest factor of trip count <= tileSize.
      uint64_t constTripCount = *mayConst;
      // Optimize for multiples of powers of 2.
      unsigned powerOf2 = llvm::countr_zero(constTripCount);
      uint64_t reducedCount = constTripCount >> powerOf2;
      if (reducedCount > tileSize) {
        reducedCount = tileSize;
        if (reducedCount % 2 != 0) {
          // Skip even; eventually tileSize >> ctz(tileSize) will be checked.
          reducedCount--;
        }
      }
      uint64_t best = 1;
      while (reducedCount > 0) {
        // Find exponent such that 2^e r <= tileSize, e <= powerOf2.
        uint64_t exponent =
            std::min(powerOf2, llvm::Log2_64(tileSize / reducedCount));
        uint64_t candidate = reducedCount << exponent;
        if (candidate > best && candidate <= tileSize &&
            constTripCount % candidate == 0) {
          best = candidate;
        }
        // We can skip even values of r.
        if (reducedCount < 2)
          break;
        reducedCount -= 2;
      }
      tileSize = best;
    }
  }
}

static void fill(SmallVectorImpl<unsigned> &tileSizes, unsigned value = 1) {
  for (unsigned &size : tileSizes)
    size = value;
}

// Gets the footprint of the intra-tile memory accesses.
static std::optional<int64_t> getInnerLoopFootprintBytes(AffineForOp forOp,
                                                         int bandSize) {
  using namespace llvm;

  Block &block = *forOp->getBlock();
  Block::iterator start(forOp);
  Block::iterator end(forOp);
  ++end;

  SmallDenseMap<Value, std::unique_ptr<MemRefRegion>, 4> regions;

  unsigned outerDepth = getNestingDepth(&*block.begin());

  // Walk this 'affine.for' operation to gather all memory regions.
  auto result = block.walk(start, end, [&](Operation *opInst) -> WalkResult {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(opInst)) {
      // Neither load nor a store op.
      return WalkResult::advance();
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    unsigned innerDepth = getNestingDepth(opInst);
    if (innerDepth < outerDepth + bandSize) {
      // Not part of the inner loop.
      return WalkResult::advance();
    }
    if (failed(region->compute(opInst,
                               /*loopDepth=*/outerDepth + bandSize))) {
      LLVM_DEBUG(opInst->emitError("error obtaining memory region"));
      return failure();
    }

    auto [it, inserted] = regions.try_emplace(region->memref);
    if (inserted) {
      it->second = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      LLVM_DEBUG(opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region"));
      return failure();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return std::nullopt;

  int64_t totalSizeInBytes = 0;
  for (const auto &region : regions) {
    std::optional<int64_t> size = region.second->getRegionSize();
    if (!size.has_value())
      return std::nullopt;
    totalSizeInBytes += *size;
  }
  return totalSizeInBytes;
}

// Returns tile sizes to use. Checks CL options; if none are specified, sets it
// based on a simple model that looks at the memory footprint and determines
// tile sizes assuming identity accesses / 1:1 tile size proportional footprint
// along each of the dimensions being tiled.
// TODO: evolve this model. Tile size determination is a large area
// to play with in general.
void AffineTilingPass::getTileSizes(ArrayRef<AffineForOp> band,
                                    SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  tileSizes->resize(band.size());

  // If the cache size is zero, set the minimum valid tile size. No good reason
  // to pick another specific size over this.
  if (cacheSizeInKiB == 0) {
    fill(*tileSizes);
    return;
  }

  // The first loop in the band.
  AffineForOp rootForOp = band[0];
  (void)rootForOp;

  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;

  // Obtain memory footprint and set tile sizes so that a tile fits in
  // the cache size. This is an approximation with the assumption that the
  // footprint increases with the tile size linearly in that dimension (i.e.,
  // assumes one-to-one access function).

  std::optional<int64_t> innerFootprint =
      getInnerLoopFootprintBytes(band[0], band.size());
  std::optional<int64_t> totalFootprint = getMemoryFootprintBytes(band[0], 0);

  if (!innerFootprint) {
    // Fill with default tile sizes if footprint is unknown.
    fill(*tileSizes, defaultTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    LLVM_DEBUG(
        rootForOp.emitWarning("memory footprint unknown: using default tile "
                              "sizes adjusted to trip count divisors"));
    return;
  }

  if (totalFootprint && *totalFootprint <= cacheSizeBytes) {
    // No need of any tiling - set tile size to 1.
    fill(*tileSizes, 1);
    return;
  }

  // For an n-d tileable band, compute the n^th root of the excess.
  double excessFactor = (double)cacheSizeBytes / *innerFootprint;
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));
  // We'll keep a running product to determine the last tile size better.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // Set last tile size to cover the balance.
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}

void AffineTilingPass::runOnOperation() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  for (AffineForOp forOp : getOperation().getOps<AffineForOp>()) {
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, forOp);
    bands.push_back(band);
  }

  // Tile each band.
  for (auto &band : bands) {
    if (!isTilingValid(band)) {
      band.front().emitRemark("tiling nest is invalid due to dependences");
      continue;
    }

    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or tileSize if one was provided.
    SmallVector<unsigned, 6> tileSizes;
    getTileSizes(band, &tileSizes);
    if (llvm::DebugFlag) {
      auto diag = band[0].emitRemark("using tile sizes [");
      for (unsigned tSize : tileSizes)
        diag << tSize << ' ';
      diag << "]\n";
    }

    if (llvm::all_of(tileSizes, [](int64_t s) { return s == 1; })) {
      LLVM_DEBUG(band.front()->emitRemark("no tiling, skipping\n"));
      continue;
    }

    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest))) {
      // An empty band always succeeds.
      assert(!band.empty() && "guaranteed to succeed on empty bands");
      LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
      continue;
    }

    // Separate full and partial tiles.
    if (separate) {
      auto intraTileLoops =
          MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
      if (failed(separateFullTiles(intraTileLoops))) {
        assert(!intraTileLoops.empty() &&
               "guaranteed to succeed on empty bands");
        LLVM_DEBUG(intraTileLoops.front()->emitRemark(
            "separation post tiling failed!\n"));
      }
    }
  }
}
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
hexagon::createAffineTilingPass(const AffineTilingOptions &options) {
  return std::make_unique<AffineTilingPass>(options);
}
