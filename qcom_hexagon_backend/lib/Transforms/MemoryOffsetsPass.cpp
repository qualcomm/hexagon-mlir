//===-- MemoryOffsets.cpp - Memory Reuse via Offset Assignment ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a function-level pass that replaces multiple
// hexagonmem.alloc operations with views into a single statically allocated
// buffer, enabling memory reuse through offset assignment.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-offsets"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_MEMORYOFFSETS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Metadata for each eligible allocation.
struct AllocInfo {
  hexagonmem::AllocOp allocOp;
  mlir::MemRefType memRefType;
  int64_t sizeInBytes;
  mlir::Location loc;
};

/// Returns static size in bytes, or -1 if dynamic or unsupported.
int64_t computeStaticSizeInBytes(mlir::MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return -1;

  auto shape = type.getShape();
  auto numElements =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return numElements * (type.getElementTypeBitWidth() / 8);
}

/// Aligns offset to the given alignment.
constexpr inline int64_t alignTo(int64_t offset, int64_t alignment) {
  return ((offset + alignment - 1) / alignment) * alignment;
}

/// Collects all static allocs in memory space 1.
llvm::SmallVector<AllocInfo, 8> collectAllocInfos(mlir::func::FuncOp func) {
  llvm::SmallVector<AllocInfo, 8> allocInfos;

  func.walk([&](hexagonmem::AllocOp allocOp) {
    auto memRefType = llvm::dyn_cast<mlir::MemRefType>(allocOp.getType());
    if (!memRefType || memRefType.getMemorySpaceAsInt() != 1)
      return;

    int64_t size = computeStaticSizeInBytes(memRefType);
    if (size > 0)
      allocInfos.push_back({allocOp, memRefType, size, allocOp.getLoc()});
  });

  return allocInfos;
}

/// Creates a 1 MB I8 buffer at function entry.
mlir::Value createGlobalBuffer(mlir::OpBuilder &builder, mlir::Location loc,
                               int64_t bufferSize, int64_t alignment) {
  auto bufferType =
      mlir::MemRefType::get({bufferSize}, builder.getI8Type(), {}, 1);
  auto alignmentAttr = builder.getI64IntegerAttr(alignment);
  return builder.create<hexagonmem::AllocOp>(loc, bufferType,
                                             mlir::ValueRange{}, alignmentAttr);
}

/// Assigns flat, aligned offsets to each alloc.
llvm::DenseMap<mlir::Operation *, int64_t>
computeFlatOffsets(llvm::SmallVectorImpl<AllocInfo> &allocInfos,
                   int64_t alignment) {
  llvm::DenseMap<mlir::Operation *, int64_t> offsetMap;
  int64_t currentOffset = 0;

  for (auto &info : allocInfos) {
    currentOffset = alignTo(currentOffset, alignment);
    offsetMap[info.allocOp] = currentOffset;
    currentOffset += info.sizeInBytes;
  }

  return offsetMap;
}

/// Replaces allocs with memref.view into the shared buffer.
void replaceAllocOpsWithViews(
    mlir::OpBuilder &builder, mlir::Value buffer,
    llvm::SmallVectorImpl<AllocInfo> &allocInfos,
    llvm::DenseMap<mlir::Operation *, int64_t> &offsetMap) {
  for (auto &info : allocInfos) {
    builder.setInsertionPoint(info.allocOp);
    auto offsetVal = builder.create<mlir::arith::ConstantIndexOp>(
        info.loc, offsetMap.lookup(info.allocOp));

    auto view = builder.create<mlir::memref::ViewOp>(
        info.loc, info.memRefType, buffer, offsetVal, mlir::ValueRange{});

    info.allocOp.getResult().replaceAllUsesWith(view);
    info.allocOp.erase();
  }
}

/// Replaces multiple allocs with views into a shared buffer.
struct MemoryOffsetsPass : public ::impl::MemoryOffsetsBase<MemoryOffsetsPass> {
  MemoryOffsetsPass() = default;
  MemoryOffsetsPass(int64_t bufferSize, int64_t alignment)
      : bufferSize(bufferSize), alignment(alignment) {}

  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    // Collect eligible allocs.
    auto allocInfos = collectAllocInfos(func);
    if (allocInfos.size() < 2)
      return;

    // Verify buffer size is sufficient
    int64_t totalRequiredSize = 0;
    for (auto &info : allocInfos) {
      totalRequiredSize = alignTo(totalRequiredSize, alignment);
      totalRequiredSize += info.sizeInBytes;
    }

    if (totalRequiredSize > bufferSize) {
      func.emitWarning() << "Required buffer size (" << totalRequiredSize
                         << " bytes) exceeds allocated buffer size ("
                         << bufferSize << " bytes)";
      return;
    }

    // Create shared buffer.
    builder.setInsertionPointToStart(&func.front());
    auto buffer =
        createGlobalBuffer(builder, func.getLoc(), bufferSize, alignment);

    // Assign offsets and replace allocs.
    auto offsetMap = computeFlatOffsets(allocInfos, alignment);
    replaceAllocOpsWithViews(builder, buffer, allocInfos, offsetMap);
  }

private:
  int64_t bufferSize = 1024 * 1024; // Default 1 MB buffer
  int64_t alignment = 2048;         // Default alignment in bytes
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createMemoryOffsetsPass() {
  return std::make_unique<MemoryOffsetsPass>();
}
