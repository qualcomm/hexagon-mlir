//===-- MemoryOffsets.cpp - VTCM Static Memory Allocator ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Replaces VTCM memref.alloc / hexagonmem.alloc ops with views into a
// shared buffer. Non-overlapping lifetimes share offsets.
//
// Lifetimes: mlir::Liveness + BufferViewFlowAnalysis (alias-aware).
// Placement: first-fit-decreasing by size.
// Fails fast if peak pressure exceeds budget.
//
// Not yet supported:
//   - Allocs inside scf.forall (concurrent regions).
//   - Spilling over-budget allocs to DDR.
//
// SPMD: assumes per-thread scratch from the wrapper.
//
// Algorithm: For each alloc, resolves all aliases (subviews, casts, iter_args)
// via BufferViewFlowAnalysis, then queries Liveness::resolveLiveness for every
// op where each alias is live — the max op-ID gives the alloc's liveEnd.
// Allocs sorted by size descending are assigned HVX-aligned (128B) offsets
// using first-fit: scan from offset 0, skip ranges occupied by already-placed
// allocs with overlapping lifetimes. Each alloc is rewritten to memref.view
// at its assigned offset; deallocs are erased.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Utils.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "hexagon/Transforms/VTCMAllocUtils.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <numeric>

#define DEBUG_TYPE "memory-offsets"

using namespace mlir;
using namespace hexagon;
using namespace hexagon::vtcm;

#define GEN_PASS_DEF_MEMORYOFFSETS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct PackingResult {
  llvm::SmallVector<Operation *> unplacedOps;
  int64_t totalSize = 0;
  int64_t maxAlignment = 0;
};

/// The function arg tagged "hexagon.scratch", or nullptr.
Value findVTCMScratchArg(func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    if (!func.getArgAttr(arg.getArgNumber(), "hexagon.scratch"))
      continue;
    auto type = dyn_cast<MemRefType>(arg.getType());
    if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
        type.getNumElements() <= 0 || !type.getElementType().isInteger(8) ||
        type.getMemorySpaceAsInt() != kVTCMMemorySpace ||
        !type.getLayout().isIdentity()) {
      func.emitWarning() << "Invalid hexagon.scratch arg type: "
                         << arg.getType();
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] scratch arg at index "
                            << arg.getArgNumber() << "\n");
    return arg;
  }
  return nullptr;
}

/// Creates a new VTCM buffer at function entry (fallback mode).
Value createFallbackBuffer(OpBuilder &builder, func::FuncOp func, int64_t size,
                           int64_t alignment) {
  builder.setInsertionPointToStart(&func.front());
  auto bufferType =
      MemRefType::get({size}, builder.getI8Type(), {}, kVTCMMemorySpace);
  auto alignmentAttr = builder.getI64IntegerAttr(alignment);
  return hexagonmem::AllocOp::create(builder, func.getLoc() << "vtcm_fallback",
                                     bufferType, ValueRange{}, alignmentAttr);
}

bool overlapsInTime(const VTCMAlloc &a, const VTCMAlloc &b) {
  return !(a.liveEnd < b.liveStart || a.liveStart > b.liveEnd);
}

bool overlapsInSpace(int64_t offset, int64_t size, const VTCMAlloc &other) {
  return offset < other.offset + other.sizeInBytes &&
         offset + size > other.offset;
}

/// Returns true if placing `candidate` at `offset` doesn't collide with any
/// already-placed alloc that's alive at the same time.
bool isFeasibleOffset(int64_t offset, const VTCMAlloc &candidate,
                      const VTCMAllocVec &allocs) {
  for (const auto &other : allocs) {
    if (other.offset < 0)
      continue;
    if (!overlapsInTime(candidate, other))
      continue;
    if (overlapsInSpace(offset, candidate.sizeInBytes, other))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Offset-finding strategies. Each takes a candidate alloc, the full alloc list
// (with already-placed entries having offset >= 0), and a budget. Returns the
// chosen offset, or -1 if the alloc can't fit.
//===----------------------------------------------------------------------===//

/// First-fit: scan from offset 0, on collision jump past the conflicting alloc.
int64_t findOffsetFirstFit(const VTCMAlloc &candidate,
                           const VTCMAllocVec &allocs, int64_t budget) {
  int64_t reqAlign = requiredAlign(candidate);
  int64_t size = candidate.sizeInBytes;
  int64_t offset = 0;
  while (offset + size <= budget) {
    bool conflict = false;
    for (const auto &other : allocs) {
      if (other.offset < 0)
        continue;
      if (!overlapsInTime(candidate, other))
        continue;
      if (overlapsInSpace(offset, size, other)) {
        offset = llvm::alignTo(other.offset + other.sizeInBytes, reqAlign);
        conflict = true;
        break;
      }
    }
    if (!conflict)
      return offset;
  }
  return -1;
}

/// Best-fit: among all feasible offsets (gaps between placed allocs), pick the
/// one that leaves the smallest remainder. Minimizes wasted space per placement
/// but can create many tiny unusable fragments over time.
int64_t findOffsetBestFit(const VTCMAlloc &candidate,
                          const VTCMAllocVec &allocs, int64_t budget) {
  int64_t reqAlign = requiredAlign(candidate);
  int64_t size = candidate.sizeInBytes;

  // Collect candidate offsets: 0, and the aligned end of every placed alloc
  // that overlaps in time.
  llvm::SmallVector<int64_t, 16> candidates;
  candidates.push_back(0);
  for (const auto &other : allocs) {
    if (other.offset < 0)
      continue;
    if (!overlapsInTime(candidate, other))
      continue;
    int64_t afterOther =
        llvm::alignTo(other.offset + other.sizeInBytes, reqAlign);
    if (afterOther + size <= budget)
      candidates.push_back(afterOther);
  }

  // Pick the feasible offset that wastes the least space (tightest fit against
  // the next occupied region or the budget ceiling).
  int64_t bestOffset = -1;
  int64_t bestWaste = std::numeric_limits<int64_t>::max();
  for (int64_t off : candidates) {
    if (off + size > budget)
      continue;
    if (!isFeasibleOffset(off, candidate, allocs))
      continue;
    int64_t nextStart = budget;
    for (const auto &other : allocs) {
      if (other.offset < 0 || !overlapsInTime(candidate, other))
        continue;
      if (other.offset >= off + size)
        nextStart = std::min(nextStart, other.offset);
    }
    int64_t waste = nextStart - (off + size);
    if (waste < bestWaste) {
      bestWaste = waste;
      bestOffset = off;
    }
  }
  return bestOffset;
}

/// Strategy function type: given a candidate, placed allocs, and budget,
/// return the offset to use (-1 if it doesn't fit).
using OffsetStrategy = int64_t (*)(const VTCMAlloc &, const VTCMAllocVec &,
                                   int64_t);

/// Runs the packing loop: sorts allocs by size descending, then calls the
/// strategy for each to assign an offset.
PackingResult packAllocations(VTCMAllocVec &allocs, int64_t budget,
                              OffsetStrategy strategy) {
  PackingResult result;
  if (allocs.empty())
    return result;

  llvm::SmallVector<size_t> order(allocs.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
    if (allocs[i].sizeInBytes != allocs[j].sizeInBytes)
      return allocs[i].sizeInBytes > allocs[j].sizeInBytes;
    return allocs[i].alignment > allocs[j].alignment;
  });

  for (size_t idx : order) {
    auto &alloc = allocs[idx];
    int64_t offset = strategy(alloc, allocs, budget);
    if (offset < 0) {
      result.unplacedOps.push_back(alloc.op);
      continue;
    }
    alloc.offset = offset;
    result.totalSize = std::max(result.totalSize, offset + alloc.sizeInBytes);
    result.maxAlignment = std::max(result.maxAlignment, requiredAlign(alloc));
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << alloc.sizeInBytes
                            << " B @ " << offset << "\n");
  }
  return result;
}

/// Replaces each placed alloc with a memref.view into the shared buffer and
/// erases the corresponding deallocs. We chase aliases to find deallocs because
/// they might operate on a cast or subview rather than the alloc result
/// directly.
void rewriteAllocOps(OpBuilder &builder, Value buffer, VTCMAllocVec &allocs,
                     BufferViewFlowAnalysis &aliases) {
  llvm::SmallPtrSet<Operation *, 8> deallocsToErase;
  for (auto &alloc : allocs)
    for (Value v : aliases.resolve(alloc.op->getResult(0)))
      for (Operation *user : v.getUsers())
        if (isa<memref::DeallocOp, hexagonmem::DeallocOp>(user))
          deallocsToErase.insert(user);

  for (auto &alloc : allocs) {
    Location loc = alloc.op->getLoc();
    builder.setInsertionPoint(alloc.op);
    auto offsetVal = arith::ConstantIndexOp::create(
        builder, loc << "vtcm_offset", alloc.offset);
    auto view = memref::ViewOp::create(builder, loc, alloc.type, buffer,
                                       offsetVal, ValueRange{});
    alloc.op->getResult(0).replaceAllUsesWith(view.getResult());
    alloc.op->erase();
    alloc.op = nullptr;
  }

  for (auto *op : deallocsToErase)
    op->erase();
}

/// Prints the offset assignment map for debugging.
void dumpOffsetMap(const VTCMAllocVec &allocs, int64_t budget, int64_t peak,
                   int64_t packedSize) {
  llvm::dbgs() << "[" DEBUG_TYPE "] === Offset Assignment Map ===\n";
  llvm::dbgs() << "[" DEBUG_TYPE "]   budget=" << budget << " peak=" << peak
               << " packed=" << packedSize << "\n";
  for (const auto &a : allocs) {
    llvm::dbgs() << "[" DEBUG_TYPE "]   " << a.type << " size=" << a.sizeInBytes
                 << " live=[" << a.liveStart << "," << a.liveEnd
                 << "] offset=" << a.offset << "\n";
  }
}

/// Attaches notes for the top-N allocs by size.
void attachTopContributorNotes(InFlightDiagnostic &diag,
                               const VTCMAllocVec &allocs, size_t maxN) {
  llvm::SmallVector<size_t> order(allocs.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
    return allocs[i].sizeInBytes > allocs[j].sizeInBytes;
  });
  for (size_t i = 0; i < std::min(maxN, order.size()); ++i) {
    const auto &alloc = allocs[order[i]];
    diag.attachNote(alloc.op->getLoc())
        << alloc.type << " (" << alloc.sizeInBytes << " B)";
  }
}

///////////////////////////////////////////////////////////////////////////////
struct MemoryOffsetsPass : public ::impl::MemoryOffsetsBase<MemoryOffsetsPass> {
  MemoryOffsetsPass() = default;
  MemoryOffsetsPass(const MemoryOffsetsOptions &options)
      : MemoryOffsetsBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexagonmem::HexagonMemDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override;
};

void MemoryOffsetsPass::runOnOperation() {
  auto func = getOperation();

  // Collect eligible allocs (static, identity-layout, VTCM).
  auto allocs = collectVTCMAllocs(func);
  if (allocs.empty())
    return;

  for (const auto &alloc : allocs) {
    if (alloc.op->getParentOfType<scf::ForallOp>()) {
      alloc.op->emitError() << "VTCM alloc in scf.forall not supported";
      return signalPassFailure();
    }
  }

  // Determine buffer — external scratch arg or self-allocated.
  Value scratchArg = findVTCMScratchArg(func);
  int64_t budget = bufferSize;
  if (scratchArg)
    budget = cast<MemRefType>(scratchArg.getType()).getNumElements();

  // Compute liveness intervals for overlap detection.
  Liveness liveness(func);
  BufferViewFlowAnalysis aliases(func);
  computeLivenessIntervals(allocs, func, liveness, aliases);
  LLVM_DEBUG(for (const auto &a
                  : allocs) {
    llvm::dbgs() << "[" DEBUG_TYPE "] " << a.sizeInBytes << " B: ["
                 << a.liveStart << ", " << a.liveEnd << "]\n";
  });

  // Early-out if naive sum of live allocs exceeds budget.
  int64_t peak = computePeakPressure(allocs);
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] peak " << peak << " B\n");
  if (peak > budget) {
    auto diag = func.emitError()
                << "VTCM peak " << peak << " > budget " << budget;
    attachTopContributorNotes(diag, allocs, /*maxN=*/5);
    return signalPassFailure();
  }

  // Pack — assign offsets using first-fit-decreasing.
  PackingResult packing = packAllocations(allocs, budget, findOffsetFirstFit);

  LLVM_DEBUG(dumpOffsetMap(allocs, budget, peak, packing.totalSize));

  if (!packing.unplacedOps.empty()) {
    auto diag = func.emitError()
                << "VTCM fragmentation: " << packing.unplacedOps.size();
    for (Operation *op : packing.unplacedOps)
      diag.attachNote(op->getLoc()) << "unplaced alloc";
    return signalPassFailure();
  }

  // Rewrite allocs → memref.view; erase deallocs.
  OpBuilder builder(func.getContext());
  Value buffer = scratchArg;
  if (!buffer) {
    int64_t maxAlign = std::max<int64_t>(alignment, packing.maxAlignment);
    buffer = createFallbackBuffer(builder, func, packing.totalSize, maxAlign);
  }
  rewriteAllocOps(builder, buffer, allocs, aliases);
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createMemoryOffsetsPass() {
  return std::make_unique<MemoryOffsetsPass>();
}
