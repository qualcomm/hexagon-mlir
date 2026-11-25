//===- DMATransferUtil.cpp - some useful general DMA functions ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Note : There is presently some overlap between here and HexagonMemToDMA.cpp
//        that will be resolved in a future refactoring.
//        Also, Rank > 2 DMA-Transfers currently generate IR level loops and
//       these need to be resolved as well.

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace hexagon {

namespace {

/// Returns true if `type` is statically shaped 2D memref.
bool isSafe(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType || memRefType.getRank() > 2)
    return false;
  return llvm::count_if(memRefType.getShape(), ShapedType::isDynamic) == 0;
}

/// Returns memory address assuming provided type is memref.
unsigned getMemorySpace(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  assert(memRefType && "expected  memref type");
  return memRefType.getMemorySpaceAsInt();
}

static Value getI32Const(Location loc, IRRewriter &rewriter, int val) {
  return arith::ConstantIndexOp::create(rewriter, loc, val);
}
} // unnamed namespace

Value createNumElements(Location loc, IRRewriter &rewriter, Value view) {
  auto viewType = view.getType();
  assert(isSafe(viewType) && "num elements cannot be created for unsafe type");
  auto memrefType = cast<MemRefType>(viewType);
  return getI32Const(loc, rewriter, memrefType.getNumElements());
}

bool canCreateDMAStart(Type sourceType, Type targetType) {
  if (!isSafe(sourceType) || !isSafe(targetType) ||
      getMemorySpace(sourceType) == getMemorySpace(targetType))
    return false;
  auto sourceMemRefType = cast<MemRefType>(sourceType);
  auto targetMemRefType = cast<MemRefType>(targetType);
  SmallVector<int64_t, 4> sourceStrides, targetStrides;
  int64_t srcOffset, targetOffset;
  if (failed(sourceMemRefType.getStridesAndOffset(sourceStrides, srcOffset)) ||
      failed(targetMemRefType.getStridesAndOffset(targetStrides, targetOffset)))
    return false;
  // Unknown memory space - don't lower
  int sourceMemSpace, targetMemSpace;
  if (!isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace) ||
      !isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace))
    return false;
  // Contiguous copies use a 1D DMA; otherwise one side must be 2D strided.
  int64_t stride, width;
  return (isContiguousMemrefType(sourceMemRefType) &&
          isContiguousMemrefType(targetMemRefType)) ||
         isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
         isStridedMultiDimMemrefType(targetMemRefType, stride, width);
}

bool isVTCMtoDDR(memref::CopyOp copyOp) {
  auto srcType = dyn_cast<MemRefType>(copyOp.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(copyOp.getTarget().getType());
  if (!srcType || !dstType)
    return false;
  return srcType.getMemorySpaceAsInt() == 1 &&
         dstType.getMemorySpaceAsInt() == 0;
}

bool createDMAStartOp(Location loc, IRRewriter &rewriter, Value source,
                      Value target, Value tagAlloc) {
  auto sourceType = source.getType();
  auto targetType = target.getType();
  if (!canCreateDMAStart(sourceType, targetType))
    return false;

  MemRefType sourceMemRefType = cast<MemRefType>(sourceType);
  MemRefType targetMemRefType = cast<MemRefType>(targetType);
  int64_t rank = sourceMemRefType.getRank();

  // Create zero index for tag access
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value, 1> tagIndex = {zero};

  SmallVector<Value, 8> zeroIndices(rank, zero);
  SmallVector<Value, 2> zeroIndex(1, zero);

  Value numElements =
      getI32Const(loc, rewriter, sourceMemRefType.getNumElements());
  if (isContiguousMemrefType(sourceMemRefType) &&
      isContiguousMemrefType(targetMemRefType)) {

    memref::DmaStartOp::create(rewriter, loc, source, zeroIndices, target,
                               zeroIndices, numElements, tagAlloc, zeroIndex);
    return true;
  }

  int64_t stride = 0, width = 0;
  if (!isStridedMultiDimMemrefType(sourceMemRefType, stride, width) &&
      !isStridedMultiDimMemrefType(targetMemRefType, stride, width))
    return false;
  memref::DmaStartOp::create(rewriter, loc, source, zeroIndices, target,
                             zeroIndices, numElements, tagAlloc, zeroIndex,
                             getI32Const(loc, rewriter, stride),
                             getI32Const(loc, rewriter, width));
  return true;
}

} // namespace hexagon
} // namespace mlir
