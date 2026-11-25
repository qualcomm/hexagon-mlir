//===- ConvertStaticAllocToAlloca.cpp - Promote small allocs to alloca ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Promotes static memref.alloc ops to memref.alloca, hoisting each alloca to
// the function entry block and erasing the paired memref.dealloc.
//
// Acceptance: allocs are sorted ascending by effective size
// (round_up(bytes, alignment)) and accepted greedily until adding the next
// candidate would push the peak simultaneous live bytes — computed via MLIR
// liveness analysis — over maxTotalAllocaSizeBytes.  Using peak rather than
// a running sum lets non-overlapping allocs share budget while correctly
// rejecting sets of simultaneously-live allocs whose combined footprint is
// too large.
//
// Coalescing: two accepted allocs of the same MemRefType share one alloca
// slot when their lifetimes are provably non-overlapping (dealloc of the
// first is seen before the second alloc in program order).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <string>

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_STATICALLOCTOALLOCA
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Returns true if `val` may escape the function, walking transitively through
// memref view-like ops that forward the underlying buffer pointer without
// consuming it. Escape sources checked:
//   - func::ReturnOp: the pointer is returned to the caller.
//   - func::CallOp: the pointer is passed to a callee that may stash it.
//   - memref::DmaStartOp / DmaWaitOp: the pointer is handed to async DMA
//     hardware which holds it beyond the op boundary.
static bool escapesFunction(Value val) {
  SmallVector<Value, 4> worklist = {val};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<func::ReturnOp>(user))
        return true;
      if (isa<func::CallOp>(user))
        return true;
      if (isa<memref::DmaStartOp, memref::DmaWaitOp>(user))
        return true;
      if (isa<memref::CastOp, memref::ReinterpretCastOp, memref::SubViewOp,
              memref::CollapseShapeOp, memref::ExpandShapeOp>(user))
        worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

struct EligibleAlloc {
  memref::AllocOp op;
  int64_t effectiveBytes;
};

// Collected output of buildDeallocMaps.
struct DeallocMaps {
  // dealloc op -> alloc op (single-dealloc allocs only; used for coalescing)
  llvm::DenseMap<Operation *, Operation *> deallocToAlloc;
  // every dealloc of every accepted alloc (collected before IR mutation)
  SmallPtrSet<Operation *, 16> allDeallocsToErase;
};

struct StaticAllocToAllocaPass
    : public ::impl::StaticAllocToAllocaBase<StaticAllocToAllocaPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticAllocToAllocaPass)

  StaticAllocToAllocaPass() = default;

  explicit StaticAllocToAllocaPass(
      const hexagon::StaticAllocToAllocaOptions &options)
      : StaticAllocToAllocaBase(options) {}

  void runOnOperation() override;

private:
  // Walk funcOp, filter eligible allocs, return sorted ascending by effective
  // size.
  SmallVector<EligibleAlloc> collectEligible(func::FuncOp funcOp);

  // Greedy liveness-aware selection within the stack budget.
  SmallVector<memref::AllocOp> selectToConvert(func::FuncOp funcOp,
                                               ArrayRef<EligibleAlloc> eligible,
                                               const Liveness &liveness);

  // Build dealloc maps needed for coalescing and erasure.
  DeallocMaps buildDeallocMaps(ArrayRef<memref::AllocOp> toConvert);

  // Assign slot IDs, create allocas, replace uses, and erase deallocs.
  void coalesceAndReplace(func::FuncOp funcOp,
                          ArrayRef<memref::AllocOp> toConvert,
                          const DeallocMaps &maps);
};

// -----------------------------------------------------------------------------
// runOnOperation
// -----------------------------------------------------------------------------

void StaticAllocToAllocaPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Collect all static, non-escaping allocs eligible for promotion,
  // sorted ascending by effective size.
  SmallVector<EligibleAlloc> eligible = collectEligible(funcOp);
  if (eligible.empty())
    return;

  // Greedily select allocs to promote, keeping peak simultaneous live
  // bytes within the configured stack budget.
  Liveness liveness(funcOp);
  SmallVector<memref::AllocOp> toConvert =
      selectToConvert(funcOp, eligible, liveness);
  if (toConvert.empty())
    return;

  // Build dealloc lookup maps needed for coalescing and safe erasure.
  DeallocMaps maps = buildDeallocMaps(toConvert);
  // Assign alloca slots, emit allocas at the entry block, replace alloc
  // uses, and erase the original allocs and their deallocs.
  coalesceAndReplace(funcOp, toConvert, maps);
}

// -----------------------------------------------------------------------------
// collectEligible
// -----------------------------------------------------------------------------

SmallVector<EligibleAlloc>
StaticAllocToAllocaPass::collectEligible(func::FuncOp funcOp) {
  SmallVector<EligibleAlloc> eligible;

  funcOp.walk([&](memref::AllocOp allocOp) {
    auto memRefType = allocOp.getType();

    if (!memRefType.hasStaticShape()) // dynamic dims cannot be stack-allocated
      return;
    if (memRefType.getMemorySpace()) // VTCM allocs (space 1) are not handled
      return;

    Type elemType = memRefType.getElementType();
    if (!elemType.isIntOrFloat()) // skip aggregates and vectors
      return;

    if (elemType.getIntOrFloatBitWidth() < 8)
      return;

    int64_t numElements = 1;
    for (int64_t dim : memRefType.getShape())
      numElements *= dim;
    int64_t totalBytes = numElements * (elemType.getIntOrFloatBitWidth() / 8);

    int64_t alignmentVal = elemType.getIntOrFloatBitWidth() / 8;
    if (auto alignAttr =
            llvm::dyn_cast_or_null<IntegerAttr>(allocOp.getAlignmentAttr()))
      if (alignAttr.getInt() > 0)
        alignmentVal = alignAttr.getInt();

    int64_t effectiveBytes = llvm::alignTo(totalBytes, alignmentVal);

    // If the alloc result escapes (returned to caller or passed to a callee
    // that may stash the pointer), promoting it to alloca would produce a
    // dangling stack pointer — exclude it from promotion.
    if (escapesFunction(allocOp.getResult()))
      return;

    // Allocs inside parallel regions (scf.forall / scf.parallel) are live
    // across concurrent iterations — promoting or coalescing them would
    // cause iterations to share stack storage and corrupt each other.
    for (Operation *parent = allocOp->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (isa<scf::ForallOp, scf::ParallelOp>(parent))
        return;
      if (isa<func::FuncOp>(parent))
        break;
    }

    // Allocs whose value is yielded out of a region (e.g. scf.if branches)
    // may be live simultaneously with allocs from the other branch after the
    // parent op — coalescing them would alias two distinct live buffers.
    for (Operation *user : allocOp->getUsers()) {
      if (isa<scf::YieldOp>(user))
        return;
    }

    eligible.push_back({allocOp, effectiveBytes});
  });

  // Sort ascending so the greedy loop in selectToConvert picks cheapest first.
  llvm::stable_sort(eligible,
                    [](const EligibleAlloc &a, const EligibleAlloc &b) {
                      return a.effectiveBytes < b.effectiveBytes;
                    });

  return eligible;
}

// -----------------------------------------------------------------------------
// selectToConvert
//
// Greedily accepts eligible allocs (cheapest-first) while the peak
// simultaneous live bytes stays within maxTotalAllocaSizeBytes.
//
// Each alloc's live interval is resolved once via MLIR liveness into a
// [startIdx, endIdx] pair using a global program-order index built by a
// single funcOp walk — O(N). Acceptance is then decided by an event-sweep
// over the accepted intervals: +bytes at startIdx, -bytes at (endIdx+1).
// The sweep runs in O(E log E) total across all candidates, avoiding the
// O(E^2 * N) cost of re-walking the function for every candidate.
// -----------------------------------------------------------------------------

SmallVector<memref::AllocOp>
StaticAllocToAllocaPass::selectToConvert(func::FuncOp funcOp,
                                         ArrayRef<EligibleAlloc> eligible,
                                         const Liveness &liveness) {
  // Assign a global program-order index to every op with a single walk — O(N).
  llvm::DenseMap<Operation *, unsigned> opIndex;
  unsigned idx = 0;
  funcOp.walk([&](Operation *op) { opIndex[op] = idx++; });
  const unsigned numOps = idx;

  // Resolve the live interval [startIdx, endIdx] for one alloc.
  // Falls back to [allocIdx, numOps-1] when liveness info is unavailable
  // for a block (conservative, matches the previous behaviour).
  auto resolveInterval =
      [&](memref::AllocOp allocOp) -> std::pair<unsigned, unsigned> {
    Value allocVal = allocOp.getResult();
    unsigned startIdx = opIndex[allocOp.getOperation()];
    unsigned endIdx = startIdx;

    funcOp.walk([&](Block *block) {
      const LivenessBlockInfo *info = liveness.getLiveness(block);
      if (!info) {
        endIdx = numOps - 1; // conservative: untracked block
        return;
      }
      bool liveInBlock =
          (block == allocOp->getBlock()) || info->isLiveIn(allocVal);
      if (!liveInBlock)
        return;

      Operation *startOp = info->getStartOperation(allocVal);
      Operation *endOp = info->getEndOperation(allocVal, startOp);
      auto sit = opIndex.find(startOp);
      auto eit = opIndex.find(endOp);
      if (sit != opIndex.end())
        startIdx = std::min(startIdx, sit->second);
      if (eit != opIndex.end())
        endIdx = std::max(endIdx, eit->second);
    });

    return {startIdx, endIdx};
  };

  // Event-sweep: compute peak live bytes from a set of (pos, delta) events.
  struct Event {
    unsigned pos;
    int64_t delta; // +bytes at interval start, -bytes at interval end+1
  };
  auto sweepPeak = [](ArrayRef<Event> events) -> int64_t {
    SmallVector<Event> sorted(events.begin(), events.end());
    llvm::sort(sorted, [](const Event &a, const Event &b) {
      return a.pos < b.pos || (a.pos == b.pos && a.delta < b.delta);
    });
    int64_t peak = 0, running = 0;
    for (const auto &ev : sorted) {
      running += ev.delta;
      if (running > peak)
        peak = running;
    }
    return peak;
  };

  // Incrementally add two events per candidate and re-sweep before committing.
  SmallVector<Event> acceptedEvents;
  SmallVector<memref::AllocOp> toConvert;

  for (const auto &e : eligible) {
    auto [startIdx, endIdx] = resolveInterval(e.op);
    acceptedEvents.push_back({startIdx, e.effectiveBytes});
    acceptedEvents.push_back({endIdx + 1, -e.effectiveBytes});
    if (sweepPeak(acceptedEvents) > maxTotalAllocaSizeBytes) {
      acceptedEvents.pop_back();
      acceptedEvents.pop_back();
      continue; // peak is not monotonic — a later non-overlapping candidate
                // may still fit even though this one didn't
    }
    toConvert.push_back(e.op);
  }

  unsigned rejected = eligible.size() - toConvert.size();
  if (!toConvert.empty() && rejected > 0)
    funcOp.emitWarning()
        << "StaticAllocToAlloca: " << rejected
        << " alloc(s) not promoted to alloca — adding each would push the"
           " peak live budget over "
        << std::to_string(maxTotalAllocaSizeBytes)
        << " B; raise --max-total-alloca-size-bytes if safe";

  return toConvert;
}

// -----------------------------------------------------------------------------
// buildDeallocMaps
//
// Two maps are built in one pass over each alloc's users:
//
//  deallocToAlloc     — used by the coalescing walk (coalesceAndReplace) to
//                       detect when a lifetime ends and return the slot to the
//                       free list. Only populated for allocs with exactly ONE
//                       dealloc: allocs with multiple deallocs (conditional
//                       branches) are excluded from coalescing to prevent
//                       premature slot reuse while the alloc is still live on
//                       another path.
//
//  allDeallocsToErase — every dealloc of every accepted alloc, collected
//                       before any IR mutation to avoid use-after-free when
//                       iterating during erasure.
// -----------------------------------------------------------------------------

DeallocMaps
StaticAllocToAllocaPass::buildDeallocMaps(ArrayRef<memref::AllocOp> toConvert) {
  DeallocMaps maps;

  for (memref::AllocOp allocOp : toConvert) {
    SmallVector<Operation *> deallocs;
    for (Operation *user : allocOp->getUsers())
      if (isa<memref::DeallocOp>(user))
        deallocs.push_back(user);

    for (Operation *d : deallocs)
      maps.allDeallocsToErase.insert(d);

    // Coalescing is only safe when there is exactly one dealloc: multiple
    // deallocs indicate the alloc is live across conditional branches, and
    // returning the slot on the first dealloc seen could allow premature
    // reuse while the alloc is still live on another control-flow path.
    if (deallocs.size() == 1)
      maps.deallocToAlloc[deallocs[0]] = allocOp.getOperation();
  }

  return maps;
}

// -----------------------------------------------------------------------------
// coalesceAndReplace
//
// Step 1 — coalescing walk: assign slot IDs.
//   Walk the function in program order. Each accepted alloc is assigned a
//   (MemRefType, slotId) pair. The slot is drawn from a per-type free list;
//   it is returned to the free list when the alloc's dealloc is encountered.
//   Two allocs of the same type share a slot (and therefore one alloca) iff
//   the first alloc's dealloc appears before the second alloc in program
//   order — i.e., their lifetimes are provably non-overlapping.
//
// Step 2 — create one alloca per unique (type, slotId) pair at the function
//   entry block. Multiple allocs assigned the same slot share a single alloca.
//
// Step 3 — replace each alloc's uses with its assigned alloca, then erase
//   the alloc op.
//
// Step 4 — erase all deallocs (collected in buildDeallocMaps before any IR
//   mutation, so all pointers remain valid).
// -----------------------------------------------------------------------------

void StaticAllocToAllocaPass::coalesceAndReplace(
    func::FuncOp funcOp, ArrayRef<memref::AllocOp> toConvert,
    const DeallocMaps &maps) {
  llvm::DenseSet<Operation *> toConvertSet;
  for (memref::AllocOp a : toConvert)
    toConvertSet.insert(a.getOperation());

  // Step 1: assign slot IDs via a program-order walk.
  llvm::DenseMap<mlir::Type, SmallVector<unsigned>> freeSlotIds;
  llvm::DenseMap<mlir::Type, unsigned> slotCounter;
  llvm::DenseMap<Operation *, unsigned> allocToSlotId;

  funcOp.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (!toConvertSet.count(op))
        return;

      mlir::Type ty = allocOp.getType();
      auto &free = freeSlotIds[ty];
      unsigned slotId = free.empty() ? slotCounter[ty]++ : free.pop_back_val();
      allocToSlotId[op] = slotId;
      return;
    }

    if (isa<memref::DeallocOp>(op)) {
      auto it = maps.deallocToAlloc.find(op);
      if (it == maps.deallocToAlloc.end())
        return;
      Operation *allocOpPtr = it->second;
      auto slotIt = allocToSlotId.find(allocOpPtr);
      if (slotIt != allocToSlotId.end()) {
        mlir::Type ty = cast<memref::AllocOp>(allocOpPtr).getType();
        // Lifetime of this alloc has ended — return slot to free list.
        freeSlotIds[ty].push_back(slotIt->second);
      }
    }
  });

  // Step 2: create one alloca per unique (type, slotId) at the entry block.
  Block *entryBlock = &funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(entryBlock);

  llvm::DenseMap<mlir::Type, SmallVector<memref::AllocaOp>> slotAllocas;
  for (auto &[ty, count] : slotCounter)
    slotAllocas[ty].resize(count); // default-initialized (null ops)

  // Iterate `toConvert` -- not `allocToSlotId` -- here and in Step 3 below.
  // `toConvert` is a stable vector fixed by collectEligible's program-order
  // walk plus a size-ascending stable_sort, so its order is reproducible
  // across otherwise-identical compilations. `allocToSlotId` is a
  // DenseMap<Operation *, unsigned>: its iteration order depends on
  // Operation* pointer hashing, i.e. on heap addresses, which vary run to
  // run (ASLR / allocator state). Creating allocas in that order made the
  // entry-block alloca order -- and therefore each alloca's SSA name --
  // non-deterministic, even though the (type, slotId) assignment itself
  // (Step 1) is already fully deterministic.
  for (memref::AllocOp allocOp : toConvert) {
    mlir::Type ty = allocOp.getType();
    unsigned slotId = allocToSlotId.lookup(allocOp.getOperation());
    auto &vec = slotAllocas[ty];
    if (!vec[slotId]) {
      vec[slotId] = memref::AllocaOp::create(
          builder, allocOp.getLoc(), allocOp.getType(),
          /*dynamicSizes=*/ValueRange{}, allocOp.getAlignmentAttr());
    }
  }

  // Step 3: replace alloc uses with their assigned alloca, then erase.
  for (memref::AllocOp allocOp : toConvert) {
    mlir::Type ty = allocOp.getType();
    unsigned slotId = allocToSlotId.lookup(allocOp.getOperation());
    memref::AllocaOp allocaOp = slotAllocas[ty][slotId];
    allocOp.getResult().replaceAllUsesWith(allocaOp.getResult());
    allocOp.erase();
  }

  // Step 4: erase deallocs.
  for (Operation *d : maps.allDeallocsToErase)
    d->erase();
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createStaticAllocToAllocaPass() {
  return std::make_unique<StaticAllocToAllocaPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createStaticAllocToAllocaPass(int64_t maxTotalAllocaSizeBytes) {
  hexagon::StaticAllocToAllocaOptions opts;
  opts.maxTotalAllocaSizeBytes = maxTotalAllocaSizeBytes;
  return std::make_unique<StaticAllocToAllocaPass>(opts);
}
