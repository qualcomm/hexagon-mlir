//====- AffineTileMemoryPass.cpp - Affine Tile Memory Pass ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements memory tiling optimization. It inserts cached copies of
// memrefs in tiled loops subject to a memory constraint.
//
//===----------------------------------------------------------------------===//

/*
 * This pass optimizes memory accesses in tiled loops by creating explicit local
 * memrefs in cache memory for each accessed memref that holds either the hyper-
 * rectangular tile accessed or the full memref. The choice of which memrefs can
 * be tiled as well as which memrefs can be promoted to full copies is chosen as
 * a knapsack problem. First, the value of caching a memref is measured by reuse
 * (ideally as the expected value, but for now we just take the product of outer
 * loop trip counts; this can be later improved by constructing a polyhedron and
 * counting the number of integer points). And the cost of caching a tile is the
 * space it takes in the knapsack; since the constraint is memory, knapsack size
 * is memory capacity and the cost of a tile is the tile size. After that, we do
 * a second knapsack to promote tiles to full copies. In this case, the value of
 * a promotion is measured by the saved memory bandwidth, which is the amount of
 * fresh data that needs to be copied each iteration. Finally, we insert code to
 * alloc/dealloc and copy to/from the cache. Special case is taken to ensure the
 * copies are not out of bounds, which can require creating a subview of dynamic
 * size if the tile is potentially out of bounds.
 */

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/AffineToLLVM/Passes.h"

#define DEBUG_TYPE "affine-tile-memory"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using mlir::func::FuncOp;
using namespace mlir::affine;
using namespace mlir::presburger;

#define GEN_PASS_DEF_AFFINETILEMEMORY
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

namespace {

enum class MemRefState {
  NotCached,
  Tiled,
  Full,
};

struct MemRefData {
  double reuse = 0;
  IntegerRelation region;
  uint64_t tileSize;
  SmallVector<int64_t, 2> tileDims;
  SmallVector<SmallVector<int64_t, 2>> tileLBs;
  SmallVector<int64_t> tileLBDivisors;
  MemRefState state;
  bool valid = true;
  SmallVector<int, 2> shift;
  uint64_t freshDataBytes;
  double sortKey;
  Value cached;
  bool hasWrite = false;

  MemRefData() : region(PresburgerSpace::getRelationSpace()) {}
};

struct AffineTileMemoryPass
    : public ::impl::AffineTileMemoryBase<AffineTileMemoryPass> {

  DominanceInfo *domInfo = nullptr;
  PostDominanceInfo *postDomInfo = nullptr;

  explicit AffineTileMemoryPass(const AffineTileMemoryOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, affine::AffineDialect,
                    memref::MemRefDialect, index::IndexDialect>();
  }

  void runOnOperation() override;
  void processLoop(SmallVectorImpl<AffineForOp> &band);
  std::function<bool(Operation *)> genDomFilterFn(Block *block);
};

void AffineTileMemoryPass::runOnOperation() {
  domInfo = &getAnalysis<DominanceInfo>();
  postDomInfo = &getAnalysis<PostDominanceInfo>();
  SmallVector<AffineForOp, 6> band;
  for (AffineForOp op : getOperation().getOps<AffineForOp>()) {
    if (op.getStep() == 1)
      continue;
    band.clear();
    getPerfectlyNestedLoops(band, op);
    // Remove intra-tile loops.
    band.erase(
        std::find_if(band.begin(), band.end(),
                     [](AffineForOp forOp) { return forOp.getStep() == 1; }),
        band.end());
    processLoop(band);
  }
}

void AffineTileMemoryPass::processLoop(SmallVectorImpl<AffineForOp> &band) {
  LLVM_DEBUG(llvm::dbgs() << "Processing loop " << band.front()
                          << " with band size " << band.size() << "\n");
  // Gather all memrefs and calculate reuse
  llvm::SmallDenseMap<Value, MemRefData> memrefData;
  llvm::SmallVector<double, 4> stack;
  AffineForOp inner = band.back();
  inner.walk([&, inner](Operation *op, const WalkStage &stage) {
    constexpr unsigned kDefaultReuse = 4;
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (forOp == inner) {
        if (stage.isBeforeAllRegions()) {
          assert(stack.empty() && "Stack should be empty on entry");
          stack.push_back(1.0);
        }
        if (stage.isAfterAllRegions()) {
          assert(stack.back() == 1.0);
          stack.pop_back();
          assert(stack.empty() && "Stack should be empty on exit");
        }
        return;
      }
      assert(!stack.empty() && "Stack should not be empty during traversal");
      uint64_t tripCount = getConstantTripCount(forOp).value_or(kDefaultReuse);
      if (stage.isBeforeAllRegions()) {
        stack.push_back(stack.back() * tripCount);
      }
      if (stage.isAfterAllRegions()) {
        double prev = stack.back();
        stack.pop_back();
        assert(stack.back() * tripCount == prev && "Stack state mismatch");
      }
    } else if (dyn_cast<AffineReadOpInterface>(op) != nullptr ||
               dyn_cast<AffineWriteOpInterface>(op) != nullptr) {
      MemRefAccess access(op);
      bool init = memrefData.count(access.memref) == 0;
      MemRefData &data = memrefData[access.memref];
      data.reuse += stack.back();
      if (isa<AffineWriteOpInterface>(op)) {
        data.hasWrite = true;
      }
      if (!init && !data.valid) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Memref " << access.memref
                   << " was marked invalid; not updating with " << op << ".\n");
        return;
      }
      if (!init && data.region.getNumInequalities() == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Memref " << access.memref
                   << " was universe; not updating with " << op << ".\n");
        return;
      }
      IntegerRelation curRel(PresburgerSpace::getRelationSpace());
      if (failed(access.getAccessRelation(curRel))) {
        // Mark as universe; can't be cached.
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to compute access relation for " << op
                   << "; setting " << access.memref << " to universe.\n");
        data.region =
            IntegerRelation::getUniverse(PresburgerSpace::getRelationSpace(
                0, access.getRank(), band.size()));
      } else {
        assert(curRel.getNumDomainVars() >= band.size() &&
               "Nested operation should have vars for tiled loops.\n");
        curRel.projectOut(band.size(), curRel.getNumDomainVars() - band.size());
        curRel.convertVarKind(VarKind::Domain, 0, band.size(), VarKind::Symbol);
        curRel.projectOut(curRel.getVarKindOffset(VarKind::Local),
                          curRel.getNumLocalVars());
        // LLVM_DEBUG({
        //   llvm::dbgs() << "Relation for " << *op << ":\n";
        //   curRel.dump();
        // });
        LLVM_DEBUG(llvm::dbgs()
                   << "Merging " << *op << " into " << access.memref << "\n");
        if (init) { // Initialize if empty
          data.region = curRel;
        }
        if (failed(data.region.unionBoundingBox(curRel))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Failed to merge access relation for " << op
                     << "; setting " << access.memref << " to universe.\n");
          data.region =
              IntegerRelation::getUniverse(PresburgerSpace::getRelationSpace(
                  0, access.getRank(), band.size()));
        }
      }
    } else if (!isa<AffineMapAccessInterface, memref::LoadOp, memref::StoreOp>(
                   op)) {
      // Non-dereferencing op; disqualify all used memrefs
      for (Value value : op->getOperands()) {
        if (isa<MemRefType>(value.getType())) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Value " << value << " used by non-dereferencing op "
                     << *op << "; marking as invalid.\n");
          memrefData[value].valid = false;
        }
      }
    }
  });
  // Disqualify aliasing memrefs. Any aliasing within a loop body disqualifies
  // both.
  AliasAnalysis &alias = getAnalysis<AliasAnalysis>();
  for (auto &entry : memrefData) {
    if (!entry.second.valid)
      continue;
    // If memref type is unranked, cannot be cached.
    MemRefType type = cast<MemRefType>(entry.first.getType());
    if (!type.hasRank()) {
      entry.second.valid = false;
      continue;
    }
    // Check aliasing.
    for (const auto &entry2 : memrefData) {
      if (entry.first != entry2.first &&
          alias.alias(entry.first, entry2.first)) {
        LLVM_DEBUG(llvm::dbgs() << "Memref " << entry.first
                                << " disqualified as it aliases with "
                                << entry2.first << ".\n");
        entry.second.valid = false;
        break;
      }
    }
  }
  // Compute tile size for each memref.
  llvm::SmallVector<Value> sortedMemRefs;
  sortedMemRefs.reserve(memrefData.size());
  for (const auto &entry : memrefData) {
    MemRefData &data = memrefData[entry.first];
    auto elementSize = getMemRefIntOrFloatEltSizeInBytes(
        cast<MemRefType>(entry.first.getType()));
    int64_t size;
    bool valid = entry.second.valid;
    if (!elementSize) {
      valid = false;
    } else {
      size = *elementSize;
    }
    const IntegerRelation &region = data.region;
    if (entry.second.valid) {
      LLVM_DEBUG({
        llvm::dbgs() << "Relation for " << entry.first << ":\n";
        region.dump();
      });
    }
    if (region.getNumConstraints() == 0) {
      valid = false;
    }
    data.tileDims.resize(region.getNumRangeVars());
    data.tileLBs.resize(region.getNumRangeVars());
    data.tileLBDivisors.resize(region.getNumRangeVars());
    SmallVector<DynamicAPInt> tmpLB;
    DynamicAPInt boundFloorDivisor;
    for (unsigned i = 0; valid && i < region.getNumRangeVars(); i++) {
      auto bound = region.getConstantBoundOnDimSize(
          region.getVarKindOffset(VarKind::Range) + i, &tmpLB,
          &boundFloorDivisor);
      if (!bound || *bound > INT64_MAX) {
        valid = false;
        break;
      }
      data.tileLBs[i].resize(tmpLB.size());
      for (int j = 0; j < tmpLB.size(); j++) {
        if (tmpLB[j] < -INT64_MAX || tmpLB[j] > INT64_MAX) {
          valid = false;
          break;
        }
        data.tileLBs[i][j] = int64_t(tmpLB[j]);
      }
      if (boundFloorDivisor < -INT64_MAX || boundFloorDivisor > INT64_MAX) {
        valid = false;
        break;
      }
      data.tileLBDivisors[i] = int64_t(boundFloorDivisor);
      LLVM_DEBUG(llvm::dbgs()
                 << "Bound for dimension " << i << ": " << *bound << "\n");
      data.tileDims[i] = int64_t(*bound);
      int64_t result;
      if (llvm::MulOverflow(size, int64_t(*bound), result)) {
        valid = false;
        break;
      }
      size = result;
    }
    data.valid = valid;
    if (valid) {
      data.tileSize = size;
      sortedMemRefs.push_back(entry.first);
    }
  }
  // Sort by decreasing reuse / tile size
  std::sort(sortedMemRefs.begin(), sortedMemRefs.end(),
            [&memrefData](Value a, Value b) {
              return memrefData[a].reuse * memrefData[b].tileSize >
                     memrefData[b].reuse * memrefData[a].tileSize;
            });
  // Greedily fit into memory
  uint64_t cacheSizeBytes =
      cacheSizeInKiB >= UINT64_MAX / 1024 ? UINT64_MAX : cacheSizeInKiB * 1024;
  uint64_t memoryUsed = 0;
  for (const auto &memref : sortedMemRefs) {
    MemRefData &data = memrefData[memref];
    uint64_t tileSize = data.tileSize;
    auto fullSize =
        getIntOrFloatMemRefSizeInBytes(cast<MemRefType>(memref.getType()));
    if (fullSize && *fullSize < tileSize) {
      tileSize = *fullSize;
    }
    if (tileSize <= cacheSizeBytes - memoryUsed) {
      if (fullSize && tileSize >= *fullSize) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Setting " << memref << "(reuse " << data.reuse << ")"
                   << " to full, using " << tileSize << " bytes.\n");
        data.state = MemRefState::Full;
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Setting " << memref << "(reuse " << data.reuse << ")"
                   << " to tiled, using " << tileSize << " bytes.\n");
        data.state = MemRefState::Tiled;
      }
      memoryUsed += tileSize;
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Not caching " << memref << "(reuse " << data.reuse << ", "
                 << tileSize << " bytes).\n");
      data.state = MemRefState::NotCached;
    }
    LLVM_DEBUG(llvm::dbgs() << "Used memory: " << memoryUsed << "/"
                            << cacheSizeBytes << " bytes.\n");
  }
  // Compute shift per innermost iteration based on tile offsets.
  for (auto &entry : memrefData) {
    MemRefData &data = entry.second;
    if (!data.valid)
      continue;
    if (data.state != MemRefState::Tiled)
      continue;
    bool valid = true;
    unsigned numRange = data.region.getNumRangeVars();
    data.shift.resize(numRange);
    for (unsigned i = 0; i < numRange; i++) {
      assert(data.tileLBs[i].size() == band.size() + 1);
      if (data.tileLBs[i][band.size() - 1] % data.tileLBDivisors[i] != 0) {
        valid = false;
      }
      data.shift[i] = data.tileLBs[i][band.size() - 1] / data.tileLBDivisors[i];
    }
    if (valid) {
      auto elementSize = getMemRefIntOrFloatEltSizeInBytes(
          cast<MemRefType>(entry.first.getType()));
      assert(elementSize &&
             "Element size should be calculable as memref is valid");
      uint64_t reuseRect = *elementSize;
      if (data.shift.size() != 0) {
        assert(data.tileDims.size() == data.shift.size() &&
               "Shift should have same size as tile dims");
        for (unsigned i = 0; i < data.tileDims.size(); i++) {
          int64_t newDim = data.tileDims[i];
          newDim -= std::abs(data.shift[i]);
          if (newDim < 0) {
            newDim = 0;
          }
          reuseRect *= newDim; // Can't overflow
        }
      } else if (data.tileDims.size() != 0) {
        reuseRect = 0;
      }
      data.freshDataBytes = data.tileSize - reuseRect;
      auto totalSize = getMemRefIntOrFloatEltSizeInBytes(
          cast<MemRefType>(entry.first.getType()));
      if (totalSize) {
        data.sortKey =
            double(data.freshDataBytes) / double(*totalSize - data.tileSize);
      } else {
        data.sortKey = -std::numeric_limits<double>::infinity();
      }
      LLVM_DEBUG(llvm::dbgs() << "Memref " << entry.first << ": "
                              << data.tileSize << " - " << reuseRect << " -> "
                              << data.freshDataBytes << " bytes loaded.\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Memref " << entry.first
                              << " does not have a valid shift.\n");
      data.shift.clear();
      auto totalSize = getMemRefIntOrFloatEltSizeInBytes(
          cast<MemRefType>(entry.first.getType()));
      if (totalSize) {
        data.freshDataBytes = *totalSize;
        data.sortKey = double(*totalSize) / double(*totalSize - data.tileSize);
      } else {
        data.sortKey = -std::numeric_limits<double>::infinity();
      }
    }
  }
  // Greedily promote by decreasing amount to be copied in each iter.
  // Reuse doesn't matter here, as every cached memref needs to be reloaded once
  // per iteration.
  std::sort(sortedMemRefs.begin(), sortedMemRefs.end(),
            [&memrefData](Value a, Value b) {
              return memrefData[a].sortKey > memrefData[b].sortKey;
            });
  for (const auto &memref : sortedMemRefs) {
    MemRefData &data = memrefData[memref];
    if (!data.valid || data.state != MemRefState::Tiled)
      continue;
    uint64_t tileSize = data.tileSize;
    auto fullSize =
        getIntOrFloatMemRefSizeInBytes(cast<MemRefType>(memref.getType()));
    if (!fullSize) {
      continue;
    }
    uint64_t extraBytes = *fullSize - tileSize;
    if (extraBytes <= cacheSizeBytes - memoryUsed) {
      LLVM_DEBUG(llvm::dbgs() << "Promoting " << memref << " to full, using "
                              << extraBytes << " extra bytes.\n");
      data.state = MemRefState::Full;
      ArrayRef<int64_t> newSize = cast<MemRefType>(memref.getType()).getShape();
      data.tileDims.assign(newSize.begin(), newSize.end());
      memoryUsed += extraBytes;
      LLVM_DEBUG(llvm::dbgs() << "Used memory: " << memoryUsed << "/"
                              << cacheSizeBytes << " bytes.\n");
    }
  }
  OpBuilder builder(band.front());

  for (auto &entry : memrefData) {
    // Generate alloc/dealloc for each cached memref.
    MemRefData &data = entry.second;
    if (!data.valid || data.state == MemRefState::NotCached)
      continue;
    Type elType = cast<MemRefType>(entry.first.getType()).getElementType();
    MemRefType type =
        MemRefType::get(data.tileDims, elType, AffineMap(), cacheMemorySpace);
    builder.setInsertionPoint(band.front());
    memref::AllocOp alloc =
        builder.create<memref::AllocOp>(band.front().getLoc(), type);
    Value cached = entry.second.cached = alloc.getResult();
    builder.setInsertionPointAfter(band.front());
    builder.create<memref::DeallocOp>(band.front().getLoc(), cached);
    if (data.state == MemRefState::Full) {
      // Replace loop body
      assert(succeeded(
          replaceAllMemRefUsesWith(entry.first, cached, band.front())));
      // Generate copies to/from cache
      builder.setInsertionPoint(band.front());
      builder.create<memref::CopyOp>(band.front().getLoc(), entry.first,
                                     cached);
      builder.setInsertionPointAfter(band.front());
      if (data.hasWrite) {
        builder.create<memref::CopyOp>(band.front().getLoc(), cached,
                                       entry.first);
        Block *block = band.back().getBody();
        auto userFilterFn = genDomFilterFn(block);
        assert(succeeded(replaceAllMemRefUsesWith(entry.first, cached, {}, {},
                                                  {}, {}, userFilterFn)));
      }
    } else {
      // Generate remap map
      SmallVector<int64_t, 5> coeffs;
      coeffs.resize(band.size() + type.getRank() + 1);
      SmallVector<AffineExpr, 2> exprs;
      exprs.reserve(type.getRank());
      assert(type.getRank() == data.tileLBs.size());
      for (int i = 0; i < type.getRank(); i++) {
        std::copy(data.tileLBs[i].begin(), data.tileLBs[i].end(),
                  coeffs.begin());
        std::fill(coeffs.begin() + type.getRank(), coeffs.end(), 0);
        coeffs[band.size() + type.getRank()] = data.tileLBs[i].back();
        AffineExpr lb =
            getAffineExprFromFlatForm(coeffs, band.size() + type.getRank(), 0,
                                      {}, band.front().getContext());
        exprs.push_back(
            getAffineDimExpr(band.size() + i, band.front().getContext()) -
            lb.floorDiv(data.tileLBDivisors[i]));
      }
      // Replace loop body
      AffineMap remap = AffineMap::get(band.size() + type.getRank(), 0, exprs,
                                       band.front().getContext());
      SmallVector<Value, 4> extraOperands;
      extraOperands.reserve(band.size());
      for (AffineForOp forOp : band) {
        extraOperands.push_back(forOp.getInductionVar());
      }
      Block *block = band.back().getBody();
      auto userFilterFn = genDomFilterFn(block);
      assert(succeeded(replaceAllMemRefUsesWith(
          entry.first, cached, {}, remap, extraOperands, {}, userFilterFn)));
      // Generate copies to/from cache
      builder.setInsertionPointToStart(band.back().getBody());

      Location loc = band.back().getLoc();
      SmallVector<OpFoldResult> subviewOffsets;
      SmallVector<OpFoldResult> subviewSizes;
      SmallVector<OpFoldResult> subviewStrides;
      SmallVector<OpFoldResult> cachedSubviewOffsets;
      SmallVector<int64_t> subviewTypeStrides;
      subviewOffsets.reserve(type.getRank());
      subviewSizes.reserve(type.getRank());
      subviewStrides.reserve(type.getRank());
      subviewTypeStrides.reserve(type.getRank());
      cachedSubviewOffsets.reserve(type.getRank());

      MemRefType srcType = cast<MemRefType>(entry.first.getType());

      // Temporary polyhedron of calculated lower bounds, given original loop
      // bounds.
      FlatAffineValueConstraints offsetSpace;
      for (AffineForOp forOp : band) {
        offsetSpace.appendDimVar(forOp.getInductionVar());
        assert(succeeded(offsetSpace.addAffineForOpDomain(forOp)));
      }
      assert(offsetSpace.getNumRangeVars() == band.size());
      offsetSpace.convertVarKind(VarKind::Range, 0, band.size(),
                                 VarKind::Symbol);
      offsetSpace.appendVar(VarKind::Range, type.getRank());
      // Append equalities for offsets.
      SmallVector<DynamicAPInt, 8> tempBound;
      tempBound.resize(offsetSpace.getNumVars() + 1);
      for (unsigned i = 0; i < type.getRank(); i++) {
        std::fill(tempBound.begin(), tempBound.end(), 0);
        assert(tempBound.size() == type.getRank() + data.tileLBs[i].size() +
                                       offsetSpace.getNumLocalVars());
        LLVM_DEBUG({
          llvm::dbgs() << "lower bound for dimension " << i << ":";
          for (auto &x : data.tileLBs[i]) {
            llvm::dbgs() << " " << x;
          }
          llvm::dbgs() << " / " << data.tileLBDivisors[i] << "\n";
        });
        // Lower bound: tileLBs[i] <= d * value + d - 1
        //              d * value - tileLBs[i] + d - 1 >= 0
        std::transform(data.tileLBs[i].begin(), data.tileLBs[i].end(),
                       tempBound.begin() + type.getRank(),
                       [](int64_t x) { return DynamicAPInt(-x); });
        // Move constant past local vars.
        std::swap(tempBound[type.getRank() + band.size()], tempBound.back());
        tempBound[i] = data.tileLBDivisors[i];
        tempBound.back() += data.tileLBDivisors[i] - 1;
        offsetSpace.addInequality(tempBound);
        tempBound.back() = 0;
        // Upper bound: d * value <= tileLBs[i]
        //              -d * value + tileLBs[i] >= 0
        std::copy(data.tileLBs[i].begin(), data.tileLBs[i].end(),
                  tempBound.begin() + type.getRank());
        // Move constant past local vars.
        std::swap(tempBound[type.getRank() + band.size()], tempBound.back());
        tempBound[i] = -data.tileLBDivisors[i];
        offsetSpace.addInequality(tempBound);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "offset space (before projecting):\n";
        offsetSpace.dump();
      });
      // Project out symbols first to preserve divisibility information.
      offsetSpace.projectOut(offsetSpace.getVarKindOffset(VarKind::Symbol),
                             offsetSpace.getNumSymbolVars());
      LLVM_DEBUG({
        llvm::dbgs() << "offset space (between projecting):\n";
        offsetSpace.dump();
      });
      offsetSpace.projectOut(offsetSpace.getVarKindOffset(VarKind::Local),
                             offsetSpace.getNumLocalVars());
      LLVM_DEBUG({
        llvm::dbgs() << "offset space (after projecting):\n";
        offsetSpace.dump();
      });

      bool isOffsetZero = true;
      bool isDynamic = false;
      SmallVector<int64_t, 2> subviewTypeSizes = data.tileDims;

      builder.setInsertionPointToStart(band.back().getBody());
      for (int i = 0; i < type.getRank(); ++i) {
        AffineExpr sumExpr =
            builder.getAffineConstantExpr(data.tileLBs[i].back());
        for (unsigned j = 0; j < band.size(); ++j) {
          sumExpr = sumExpr + builder.getAffineDimExpr(j) * data.tileLBs[i][j];
        }
        AffineMap lbMap = AffineMap::get(band.size(), 0, sumExpr);
        Value offset =
            builder.create<affine::AffineApplyOp>(loc, lbMap, extraOperands)
                .getResult();
        OpFoldResult copySize = builder.getIndexAttr(data.tileDims[i]);
        OpFoldResult cachedOffset = builder.getIndexAttr(0);

        // Compute lower and upper bounds on offset.
        IntegerRelation temp(offsetSpace);
        temp.projectOut(0, i);
        temp.projectOut(1, temp.getNumVars() - 1);
        auto lb = temp.getConstantBound(BoundType::LB, 0),
             ub = temp.getConstantBound(BoundType::UB, 0);

        if (!lb || *lb < 0) {
          LLVM_DEBUG(
              if (lb) {
                llvm::dbgs() << "lower bound for dimension " << i << " is "
                             << *lb << ", which is less than zero.\n";
              } else {
                llvm::dbgs()
                    << "could not find lower bound for dimension " << i << "\n";
              });
          isDynamic = true;
          isOffsetZero = false;
          subviewTypeSizes[i] = ShapedType::kDynamic;
          // Need to compute max with 0.
          Value zero =
              builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
          offset = builder.create<index::MaxSOp>(loc, zero, offset);
          assert(isa<IntegerAttr>(cast<Attribute>(copySize)) &&
                 "copySize should be a constant here");
          Value copySizeValue = builder.create<arith::ConstantOp>(
              loc, cast<IntegerAttr>(cast<Attribute>(copySize)));
          cachedOffset =
              builder.create<index::SubOp>(loc, offset, zero).getResult();
          // size = size - (max(0, offset) - offset)
          copySize = builder
                         .create<index::SubOp>(loc, copySizeValue,
                                               cast<Value>(cachedOffset))
                         .getResult();
        }
        if (srcType.isDynamicDim(i) || !ub ||
            *ub + data.tileDims[i] > srcType.getDimSize(i)) {
          LLVM_DEBUG(
              if (srcType.isDynamicDim(i)) {
                llvm::dbgs() << "src dimension " << i << " has dynamic size.\n";
              } if (ub) {
                llvm::dbgs() << "upper bound for dimension " << i << " is "
                             << *ub << " + " << data.tileDims[i]
                             << ", which is greater than dimension size ";
                if (srcType.isDynamicDim(i)) {
                  llvm::dbgs() << "(dynamic)";
                } else {
                  llvm::dbgs() << srcType.getDimSize(i);
                }
                llvm::dbgs() << ".\n";
              } else {
                llvm::dbgs()
                    << "could not find upper bound for dimension " << i << "\n";
              });
          isDynamic = true;
          subviewTypeSizes[i] = ShapedType::kDynamic;
          Value copySizeValue;
          if (isa<Value>(copySize)) {
            copySizeValue = cast<Value>(copySize);
          } else {
            copySizeValue = builder.create<arith::ConstantOp>(
                loc, cast<IntegerAttr>(cast<Attribute>(copySize)));
          }
          Value dimSize = builder.create<memref::DimOp>(loc, entry.first, i);
          copySize = builder
                         .create<index::MinSOp>(
                             loc, copySizeValue,
                             builder.create<index::SubOp>(loc, dimSize, offset))
                         .getResult();
        }

        subviewOffsets.push_back(offset);
        subviewSizes.push_back(copySize);
        subviewStrides.push_back(OpFoldResult(builder.getIndexAttr(1)));
        cachedSubviewOffsets.push_back(cachedOffset);
      }

      // Compute subview type strides
      int64_t srcOffset; // Ignored
      if (failed(srcType.getStridesAndOffset(subviewTypeStrides, srcOffset))) {
        subviewTypeStrides.resize(type.getRank());
        for (int i = 0; i < type.getRank(); i++) {
          subviewTypeStrides[i] = ShapedType::kDynamic;
        }
      }

      MemRefType subviewType = MemRefType::get(
          subviewTypeSizes, elType,
          StridedLayoutAttr::get(band.front().getContext(),
                                 ShapedType::kDynamic, subviewTypeStrides));

      // Create subview of the original memref
      Value originalMemRefSubView = builder.create<memref::SubViewOp>(
          loc, subviewType, entry.first, subviewOffsets, subviewSizes,
          subviewStrides);

      // If any offset or size is dynamic, need to create subview of cached.
      Value cachedSubview = cached;
      if (isDynamic) {
        SmallVector<int64_t> cachedStrides;
        assert(succeeded(type.getStridesAndOffset(cachedStrides, srcOffset)));
        int64_t offsetValue = isOffsetZero ? 0 : ShapedType::kDynamic;
        MemRefType cachedSubviewType =
            MemRefType::get(subviewTypeSizes, elType,
                            StridedLayoutAttr::get(band.front().getContext(),
                                                   offsetValue, cachedStrides));
        cachedSubview = builder.create<memref::SubViewOp>(
            loc, cachedSubviewType, cached, cachedSubviewOffsets, subviewSizes,
            subviewStrides);
      }

      // Copy from original memref subview to cache
      builder.create<memref::CopyOp>(loc, originalMemRefSubView, cachedSubview);
      builder.setInsertionPoint(band.back().getBody()->getTerminator());
      if (data.hasWrite) {
        builder.create<memref::CopyOp>(loc, cachedSubview,
                                       originalMemRefSubView);
      }
    }
  }
}

// Generates a predicate lambda that, supplied an op, returns true if and only
// if:
//   - The first op in the given block dominates the supplied op
//   - The last op in the given block post-dominates the supplied op
std::function<bool(Operation *)>
AffineTileMemoryPass::genDomFilterFn(Block *block) {
  Operation *firstOp = &block->front();
  Operation *lastOp = &block->back();

  return [this, firstOp, lastOp](Operation *op) {
    if (!domInfo->dominates(firstOp, op))
      return false;
    if (!postDomInfo->postDominates(lastOp, op))
      return false;
    return true;
  };
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
hexagon::createAffineTileMemoryPass(const AffineTileMemoryOptions &options) {
  return std::make_unique<AffineTileMemoryPass>(options);
}
