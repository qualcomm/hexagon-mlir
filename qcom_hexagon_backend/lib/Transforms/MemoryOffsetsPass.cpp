//===-- MemoryOffsets.cpp - Memory Reuse via Offset Assignment ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a function-level pass that replaces alloc operations
// (memref.alloc/hexagonmem.alloc) in VTCM address space (space 1) with views
// into a single buffer, enabling memory reuse through offset assignment.
//
// If a function argument marked with the "hexagon.scratch" attribute is
// present, that argument is used as the shared buffer (external VTCM mode).
// Otherwise, a single hexagonmem.alloc is created internally (fallback mode).
//
// In both modes, dealloc ops corresponding to replaced allocs are removed.
//
// External VTCM mode uses static offsets into the scratch buffer. SPMD safety
// is achieved by the wrapper allocating per_instance_scratch * prod(grid) bytes
// and passing each thread its own non-overlapping slice of that allocation.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

#define DEBUG_TYPE "memory-offsets"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_MEMORYOFFSETS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Metadata for each eligible allocation.
/// allocOp may be either a memref::AllocOp or a hexagonmem::AllocOp.
struct AllocInfo {
  mlir::Operation *allocOp;
  mlir::MemRefType memRefType;
  int64_t sizeInBytes;
};

/// Returns static size in bytes, or -1 if dynamic or unsupported.
int64_t computeStaticSizeInBytes(mlir::MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return -1;

  unsigned bitWidth = type.getElementTypeBitWidth();
  if (bitWidth % 8 != 0)
    return -1; // Sub-byte element types (e.g. i4, i1) not supported.

  auto shape = type.getShape();
  auto numElements = std::accumulate(shape.begin(), shape.end(), int64_t(1),
                                     std::multiplies<int64_t>());
  return numElements * static_cast<int64_t>(bitWidth / 8);
}

/// Collects all static memref.alloc and hexagonmem.alloc ops in VTCM
/// address space (space 1), preserving program order via a single walk.
llvm::SmallVector<AllocInfo, 8> collectAllocInfos(mlir::func::FuncOp func) {
  llvm::SmallVector<AllocInfo, 8> allocInfos;

  auto tryCollect = [&](mlir::Operation *op, mlir::MemRefType memRefType) {
    if (memRefType.getMemorySpaceAsInt() != 1)
      return;
    if (!memRefType.getLayout().isIdentity())
      return;
    int64_t size = computeStaticSizeInBytes(memRefType);
    if (size > 0)
      allocInfos.push_back({op, memRefType, size});
  };

  // Single walk to preserve source order for deterministic offset assignment.
  // Both op types implement AllocationOpInterface, so a single walk dispatches
  // on the interface.
  func.walk([&](mlir::Operation *op) {
    if (!isa<memref::AllocOp, hexagonmem::AllocOp>(op))
      return;
    if (auto memRefType =
            llvm::dyn_cast<mlir::MemRefType>(op->getResult(0).getType()))
      tryCollect(op, memRefType);
  });

  return allocInfos;
}

/// Returns true if the type is compatible with VTCM scratch usage:
/// rank-1 memref of i8 in memory space 1 with identity layout.
bool isCompatibleVTCMScratchType(mlir::Type type) {
  auto memRefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memRefType)
    return false;
  return memRefType.getRank() == 1 &&
         memRefType.getElementType().isInteger(8) &&
         memRefType.getMemorySpaceAsInt() == 1 &&
         memRefType.getLayout().isIdentity();
}

/// Scans function arguments for one marked with "hexagon.scratch".
/// Validates the argument type; emits a warning and returns nullptr if invalid.
/// Returns the argument value if found and valid, nullptr otherwise.
mlir::Value findVTCMScratchArg(mlir::func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    if (!func.getArgAttr(arg.getArgNumber(), "hexagon.scratch"))
      continue;

    if (!isCompatibleVTCMScratchType(arg.getType())) {
      func.emitWarning()
          << "Ignoring invalid hexagon.scratch argument type; expected "
             "rank-1 memref of i8 in memory space 1, got "
          << arg.getType();
      return nullptr;
    }

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Using external VTCM scratch "
                               "arg at index "
                            << arg.getArgNumber() << "\n");
    return arg;
  }
  return nullptr;
}

/// Creates an i8 buffer of exactly requiredSize bytes at function entry
/// (fallback mode).
mlir::Value createFallbackBuffer(mlir::OpBuilder &builder, mlir::Location loc,
                                 int64_t requiredSize, int64_t alignment) {
  auto bufferType =
      mlir::MemRefType::get({requiredSize}, builder.getI8Type(), {}, 1);
  auto alignmentAttr = builder.getI64IntegerAttr(alignment);
  return hexagonmem::AllocOp::create(builder, loc, bufferType,
                                     mlir::ValueRange{}, alignmentAttr);
}

/// Assigns flat, aligned offsets to each alloc.
///
/// Current restrictions:
///  - All allocs are laid out simultaneously (no lifetime/liveness analysis).
///    Non-overlapping lifetimes could share memory (future work).
///  - Allocations must not be returned from loops.
///  - Conditional allocations are always reserved (conservative).
///  - Buffer base and inter-view alignment share the same value (default 2048).
llvm::DenseMap<mlir::Operation *, int64_t>
computeFlatOffsets(llvm::SmallVectorImpl<AllocInfo> &allocInfos,
                   int64_t alignment) {
  llvm::DenseMap<mlir::Operation *, int64_t> offsetMap;
  int64_t currentOffset = 0;

  for (auto &info : allocInfos) {
    currentOffset = llvm::alignTo(currentOffset, alignment);
    offsetMap[info.allocOp] = currentOffset;
    currentOffset += info.sizeInBytes;
  }

  return offsetMap;
}

/// Replaces allocs with memref.view into the shared buffer using static
/// offsets. Used in fallback mode and current external mode.
void replaceAllocOpsWithViews(
    mlir::OpBuilder &builder, mlir::Value buffer,
    llvm::SmallVectorImpl<AllocInfo> &allocInfos,
    llvm::DenseMap<mlir::Operation *, int64_t> &offsetMap) {
  for (auto &info : allocInfos) {
    builder.setInsertionPoint(info.allocOp);
    auto offsetVal = mlir::arith::ConstantIndexOp::create(
        builder, info.allocOp->getLoc(), offsetMap.lookup(info.allocOp));

    auto view = mlir::memref::ViewOp::create(builder, info.allocOp->getLoc(),
                                             info.memRefType, buffer, offsetVal,
                                             mlir::ValueRange{});

    info.allocOp->getResult(0).replaceAllUsesWith(view);
    info.allocOp->erase();
  }
}

/// Removes deallocs only for allocations replaced by this pass.
void removeDeallocOpsForReplacedAllocs(
    llvm::SmallVectorImpl<AllocInfo> &allocInfos) {
  llvm::SmallVector<mlir::Operation *> toErase;
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;

  for (auto &info : allocInfos) {
    for (mlir::Operation *user : info.allocOp->getUsers()) {
      if (isa<memref::DeallocOp, hexagonmem::DeallocOp>(user) &&
          seen.insert(user).second)
        toErase.push_back(user);
    }
  }

  for (auto op : toErase)
    op->erase();
}

/// Replaces multiple allocs with views into a shared buffer.
struct MemoryOffsetsPass : public ::impl::MemoryOffsetsBase<MemoryOffsetsPass> {
  MemoryOffsetsPass() = default;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hexagonmem::HexagonMemDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());
    // bufferSize and alignment have tablegen-specified defaults (1048576 and
    // 2048 respectively), so they are always positive when invoked through the
    // normal pass pipeline.

    // Collect eligible alloc ops in VTCM address space.
    auto allocInfos = collectAllocInfos(func);
    if (allocInfos.empty())
      return;

    // Compute total required size from static allocs (naive summation —
    // no liveness analysis; see restrictions in computeFlatOffsets).
    int64_t totalRequiredSize = 0;
    for (auto &info : allocInfos) {
      totalRequiredSize = llvm::alignTo(totalRequiredSize, alignment);
      totalRequiredSize += info.sizeInBytes;
    }

    // Determine buffer: use existing VTCM scratch arg if present,
    // otherwise create an internal hexagonmem.alloc (fallback).
    mlir::Value buffer = findVTCMScratchArg(func);
    if (!buffer) {
      // Fallback: check against configured budget before allocating.
      if (totalRequiredSize > bufferSize) {
        func.emitError() << "Required VTCM size (" << totalRequiredSize
                         << " bytes) exceeds configured budget ("
                         << static_cast<int64_t>(bufferSize)
                         << " bytes). Increase budget or reduce kernel "
                            "allocations.";
        return signalPassFailure();
      }
      // Allocate exactly what is needed, not the full budget cap.
      builder.setInsertionPointToStart(&func.front());
      buffer = createFallbackBuffer(builder, func.getLoc(), totalRequiredSize,
                                    alignment);

      // Fallback mode: static offsets.
      // removeDeallocOps must precede replaceAllocOpsWithViews: the former
      // iterates allocOp->getUsers(), which becomes invalid after the latter
      // erases the alloc ops.
      auto offsetMap = computeFlatOffsets(allocInfos, alignment);
      removeDeallocOpsForReplacedAllocs(allocInfos);
      replaceAllocOpsWithViews(builder, buffer, allocInfos, offsetMap);
      return;
    }

    // External VTCM mode: static offsets into provided scratch argument.
    //
    // The scratch argument must be statically sized (memref<Nxi8, 1>) so
    // the pass can validate at compile time that all allocations fit.
    // InsertScratchArgPass always produces a static type; this check is only
    // reachable when hexagon.scratch is written by hand in MLIR source.
    auto scratchType = llvm::cast<mlir::MemRefType>(buffer.getType());
    if (!scratchType.hasStaticShape()) {
      func.emitError()
          << "hexagon.scratch argument must have a static size, got "
          << scratchType;
      return signalPassFailure();
    }

    int64_t scratchSizeBytes = scratchType.getNumElements();
    if (scratchSizeBytes <= 0) {
      func.emitError()
          << "hexagon.scratch argument must have a positive size, got "
          << scratchSizeBytes << " bytes";
      return signalPassFailure();
    }

    if (totalRequiredSize > scratchSizeBytes) {
      func.emitError()
          << "Required VTCM size (" << totalRequiredSize
          << " bytes) exceeds per-instance VTCM budget (" << scratchSizeBytes
          << " bytes). Reduce kernel VTCM usage, reduce grid size, or "
             "increase total scratch allocation.";
      return signalPassFailure();
    }

    // removeDeallocOps must precede replaceAllocOpsWithViews: the former
    // iterates allocOp->getUsers(), which becomes invalid after the latter
    // erases the alloc ops.
    auto offsetMap = computeFlatOffsets(allocInfos, alignment);
    removeDeallocOpsForReplacedAllocs(allocInfos);
    replaceAllocOpsWithViews(builder, buffer, allocInfos, offsetMap);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createMemoryOffsetsPass() {
  return std::make_unique<MemoryOffsetsPass>();
}
