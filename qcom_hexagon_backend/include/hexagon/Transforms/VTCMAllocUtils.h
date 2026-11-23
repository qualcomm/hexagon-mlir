//===-- VTCMAllocUtils.h - Shared VTCM alloc analysis helpers -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Internal header shared between MemoryOffsetsPass and related VTCM passes.
// Not part of the public Transforms API.
//
// Provides:
//   - VTCMAlloc / VTCMAllocVec  — per-alloc descriptor with liveness fields
//   - collectVTCMAllocs         — walk a func, collect eligible VTCM allocs
//   - computeLivenessIntervals  — fill liveStart/liveEnd using Liveness +
//                                 BufferViewFlowAnalysis
//   - computePeakPressure       — sweep-line peak of simultaneously-live bytes
//
// Packing helpers (findFirstFitOffset, packAllocations, rewriteAllocOps) are
// NOT shared; they live exclusively in MemoryOffsetsPass.cpp.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlir {
namespace hexagon {
namespace vtcm {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum alignment for any VTCM alloc (HVX vector width).
inline constexpr int64_t kHVXVectorAlignment = 128;

/// MLIR address space for VTCM memory.
inline constexpr int64_t kVTCMMemorySpace = 1;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Per-alloc descriptor produced by collectVTCMAllocs.
/// liveStart / liveEnd are populated by computeLivenessIntervals.
/// offset is populated by the packing algorithm (MemoryOffsetsPass only).
struct VTCMAlloc {
  Operation *op;
  MemRefType type;
  int64_t sizeInBytes;
  int64_t alignment; ///< 0 = no explicit alignment; kHVXVectorAlignment applied
  int liveStart = 0;
  int liveEnd = 0;
  int64_t offset = -1; ///< -1 = not yet placed
};

using VTCMAllocVec = llvm::SmallVectorImpl<VTCMAlloc>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the byte size of a static int/float memref, or -1 if unsupported.
inline int64_t computeStaticSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return -1;
  unsigned bitWidth = type.getElementTypeBitWidth();
  int64_t bytesPerElem;
  if (bitWidth < 8) {
    // Sub-byte scalar types (e.g. i1) are not bit-packed by the
    // memref-to-LLVM lowering: its sizeof(elem) idiom (GEP over a null
    // pointer) yields the element's ABI store size, which is a full byte for
    // i1 on this target. Treating these as size 0 here would make
    // collectVTCMAllocs() silently skip the alloc, leaving it out of the
    // scratch-arena offset plan while the lowering still allocates real,
    // uncoordinated (malloc'd) storage for it -- which can then alias a
    // buffer that *was* placed in the scratch arena.
    bytesPerElem = 1;
  } else if (bitWidth % 8 == 0) {
    bytesPerElem = static_cast<int64_t>(bitWidth / 8);
  } else {
    return -1;
  }
  return type.getNumElements() * bytesPerElem;
}

/// Walk `func` and collect every static, identity-layout VTCM alloc
/// (memref.alloc or hexagonmem.alloc in address space 1) that is eligible for
/// offset planning.  Returns them in program order.
inline llvm::SmallVector<VTCMAlloc, 8> collectVTCMAllocs(func::FuncOp func) {
  llvm::SmallVector<VTCMAlloc, 8> result;
  func.walk([&](Operation *op) {
    if (!isa<memref::AllocOp, hexagonmem::AllocOp>(op))
      return;
    auto type = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (!type || type.getMemorySpaceAsInt() != kVTCMMemorySpace)
      return;
    if (!type.getLayout().isIdentity()) {
      op->emitWarning("unsafe VTCM alloc detected (non-identity layout); "
                      "excluding from offset planning");
      return;
    }
    int64_t size = computeStaticSizeInBytes(type);
    if (size <= 0) {
      op->emitWarning("unsafe VTCM alloc detected (dynamic or unsupported "
                      "size); excluding from offset planning");
      return;
    }
    // Explicit alignment attribute (set by makeVtcmAlloc); 0 = let packer
    // apply kHVXVectorAlignment floor.
    int64_t align = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("alignment"))
      align = a.getInt();
    else if (auto h = dyn_cast<hexagonmem::AllocOp>(op))
      align = h.getAlignment();
    result.push_back({op, type, size, align});
  });
  return result;
}

/// Fill liveStart / liveEnd on each alloc using alias-aware liveness.
///
/// An alloc's lifetime is not limited to its direct users: a subview, cast,
/// or iter_arg of the alloc result keeps the underlying buffer live until the
/// last use of any derived value.  BufferViewFlowAnalysis::resolve() enumerates
/// all such transitive aliases; Liveness::resolveLiveness(v) then returns every
/// op where each alias is live.  The maximum pre-order id across all those ops
/// becomes liveEnd, ensuring that overlapping allocs are never assigned the
/// same VTCM offset by the packing step.
///
/// liveStart = pre-order id of the alloc op itself.
/// liveEnd   = max pre-order id of all ops that transitively use the alloc's
///             result (following memref aliases tracked by
///             BufferViewFlowAnalysis).
inline void computeLivenessIntervals(VTCMAllocVec &allocs, func::FuncOp func,
                                     const Liveness &liveness,
                                     BufferViewFlowAnalysis &aliases) {
  // Number ops in pre-order: parent before children, siblings in program order.
  llvm::DenseMap<Operation *, int> opToId;
  int currentId = 0;
  func.walk<WalkOrder::PreOrder>(
      [&](Operation *op) { opToId[op] = currentId++; });

  for (auto &alloc : allocs) {
    int startId = opToId.lookup(alloc.op);
    int endId = startId;
    for (Value v : aliases.resolve(alloc.op->getResult(0))) {
      for (Operation *op : liveness.resolveLiveness(v)) {
        auto it = opToId.find(op);
        if (it != opToId.end())
          endId = std::max(endId, it->second);
      }
    }
    alloc.liveStart = startId;
    alloc.liveEnd = endId;
  }
}

/// Sweep-line over alloc/free events to find the maximum number of bytes
/// simultaneously live — a quick lower bound on how much contiguous VTCM is
/// needed.  Allocs with non-overlapping live ranges do NOT contribute to the
/// same peak instant.
///
/// If the result already exceeds the budget, no packing attempt is needed.
///
/// Note: this ignores alignment padding between allocs, so the packing step
/// can still fail even when peak ≤ budget.  Use this as a fast pre-check only.
inline int64_t computePeakPressure(const VTCMAllocVec &allocs) {
  struct Event {
    int time;
    int64_t sizeDelta;
  };
  llvm::SmallVector<Event, 16> events;
  events.reserve(allocs.size() * 2);
  for (const auto &a : allocs) {
    events.push_back({a.liveStart, a.sizeInBytes});
    events.push_back({a.liveEnd + 1, -a.sizeInBytes});
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const Event &a, const Event &b) {
                     if (a.time == b.time)
                       return a.sizeDelta < b.sizeDelta; // frees before allocs
                     return a.time < b.time;
                   });
  int64_t current = 0, peak = 0;
  for (const auto &e : events) {
    current += e.sizeDelta;
    peak = std::max(peak, current);
  }
  return peak;
}

/// Effective byte alignment for a VTCM alloc: the larger of kHVXVectorAlignment
/// and the alloc's own requested alignment.
inline int64_t requiredAlign(const VTCMAlloc &a) {
  return std::max<int64_t>(kHVXVectorAlignment, a.alignment);
}

} // namespace vtcm
} // namespace hexagon
} // namespace mlir
