//===- HexmemCpyToDMA.cpp - hexagonmem.copy to memref.dma* conversion ====-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting of hexagonmem.copy ops to memref.dma* ops
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexmem-cpy-to-dma"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) (DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXMEMCPYTODMA
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

using CandidateTy = mlir::hexagonmem::CopyOp;
using CandidatesTy = SmallVector<CandidateTy>;

struct HexmemCpyToDMAPass
    : public ::impl::HexmemCpyToDMABase<HexmemCpyToDMAPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override;
};

static Value getI32Const(Location loc, IRRewriter &rewriter, int val) {
  return rewriter.create<arith::ConstantIndexOp>(loc, val);
}

static bool isValidCandidate(CandidateTy op) {
  auto sourceType = op.getSource().getType();
  auto targetType = op.getTarget().getType();

  auto sourceMemRefType = dyn_cast<MemRefType>(sourceType);
  auto targetMemRefType = dyn_cast<MemRefType>(targetType);

  if (!sourceMemRefType || !targetMemRefType)
    return false;

  auto isContiguousOrMultiDimMemrefType = [&](MemRefType type) {
    int64_t stride, width;
    return isContiguousMemrefType(type) ||
           isStridedMultiDimMemrefType(type, stride, width);
  };

  // If any of source/target is not (multiDim-strided or contiguous), don't
  // lower
  if (!isContiguousOrMultiDimMemrefType(sourceMemRefType) ||
      !isContiguousOrMultiDimMemrefType(targetMemRefType))
    return false;

  int sourceMemSpace, targetMemSpace;
  // Unknown memory space - don't lower
  if (!isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace) ||
      !isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace))
    return false;

  if (targetMemSpace == sourceMemSpace)
    return false;

  // if both are strided don't lower
  int64_t stride, width;
  if ((isStridedMultiDimMemrefType(sourceMemRefType, stride, width)) &&
      (isStridedMultiDimMemrefType(targetMemRefType, stride, width)))
    return false;

  // Skip DMA conversion for strided multi-dim memrefs unless
  //  transferring DDR->VTCM (source) or VTCM->DDR (target)
  if ((isStridedMultiDimMemrefType(sourceMemRefType, stride, width) &&
       !(sourceMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
         targetMemSpace == VTCM_ADDRESS_SPACE)) ||
      (isStridedMultiDimMemrefType(targetMemRefType, stride, width) &&
       !(targetMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
         sourceMemSpace == VTCM_ADDRESS_SPACE)))
    return false;

  // Skip lowering for non-static
  if (!sourceMemRefType.hasStaticShape() || !targetMemRefType.hasStaticShape())
    return false;

  // All of them hold:
  // both are either contiguous or strided memrefs
  // both addres-spaces are int
  // source not strided, or (sourceMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
  // targetMemSpace == VTCM_ADDRESS_SPACE) target not strided, or
  // (targetMemSpace == DEFAULT_DDR_ADDRESS_SPACE && sourceMemSpace ==
  // VTCM_ADDRESS_SPACE)

  return true;
}

void populateCandidate(mlir::hexagonmem::CopyOp op, CandidatesTy &candidates) {
  if (isValidCandidate(op)) {
    DBG("selected candidate: " << op);
    candidates.emplace_back(op);
  }
}

void replaceWithDMA(mlir::hexagonmem::CopyOp copyOp, FunctionOpInterface funcOp,
                    IRRewriter &rewriter) {
  rewriter.setInsertionPoint(copyOp.getOperation());
  auto loc = copyOp->getLoc();
  auto source = copyOp.getSource();
  auto target = copyOp.getTarget();

  auto sourceType = source.getType();
  auto targetType = target.getType();
  assert(isa<MemRefType>(sourceType) && isa<MemRefType>(targetType));

  // Allocate a tag buffer for DMA operations
  auto tagType = MemRefType::get({1}, rewriter.getI32Type());
  Value tagAlloc = rewriter.create<memref::AllocOp>(loc, tagType);

  // Create zero index for tag access
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 1> tagIndex = {zero};

  MemRefType sourceMemRefType = cast<MemRefType>(sourceType);
  MemRefType targetMemRefType = cast<MemRefType>(targetType);
  int64_t rank = sourceMemRefType.getRank();
  SmallVector<Value, 8> zeroIndices(rank, getI32Const(loc, rewriter, 0));
  SmallVector<Value, 8> tileIndices(rank, getI32Const(loc, rewriter, 0));
  SmallVector<Value, 2> zeroIndex(1, getI32Const(loc, rewriter, 0));

  // Extract strides and offsets for source and target memrefs
  SmallVector<int64_t, 4> sourceStrides, targetStrides;
  int64_t srcOffset, targetOffset;
  if (failed(sourceMemRefType.getStridesAndOffset(sourceStrides, srcOffset)) ||
      failed(targetMemRefType.getStridesAndOffset(targetStrides, targetOffset)))
    return;

  auto emitDMA = [loc, &rewriter, &tagAlloc, &tagIndex](
                     Value src, Value dst, SmallVector<Value, 8> srcIndices,
                     SmallVector<Value, 8> destIndices, Value numElements,
                     Value stride, Value width) {
    rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, destIndices,
                                        numElements, tagAlloc, tagIndex, stride,
                                        width);
    rewriter.create<memref::DmaWaitOp>(loc, tagAlloc, tagIndex, numElements);
  };

  int sourceMemSpace, targetMemSpace;
  // Unknown memory space - don't lower
  if (!(isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace)) ||
      !(isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace)))
    return;

  if ((rank > 2) && (!isContiguousMemrefType(sourceMemRefType) ||
                     !isContiguousMemrefType(targetMemRefType))) {
    // Generate nested loops for dimensions beyond the last two
    SmallVector<Value, 8> loopIvs;
    scf::ForOp outerLoop;
    scf::ForOp curLoop;

    // Create inner loops for the remaining (rank - 2) dimensions
    for (int64_t i = 0; i < rank - 2; ++i) {
      curLoop = rewriter.create<scf::ForOp>(
          loc, getI32Const(loc, rewriter, 0),
          getI32Const(
              loc, rewriter,
              isa<MemRefType>(copyOp.getSource().getType())
                  ? cast<MemRefType>(copyOp.getSource().getType()).getDimSize(i)
                  : sourceMemRefType.getDimSize(i)),
          getI32Const(loc, rewriter, 1));
      rewriter.setInsertionPointToStart(curLoop.getBody());
      loopIvs.push_back(curLoop.getInductionVar());

      if (i == 0) {
        outerLoop = curLoop; // The first loop is the outermost loop
      }
    }

    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    for (int64_t i = 0; i < loopIvs.size(); ++i) {
      offsets[i] = loopIvs[i];
      tileIndices[i] = loopIvs[i];
    }

    // if transfer from DDR to VTCM
    // (sourceMemSpace == DEFAULT_DDR_ADDRESS_SPACE && targetMemSpace ==
    // VTCM_ADDRESS_SPACE)
    if (sourceMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
        targetMemSpace == VTCM_ADDRESS_SPACE) {
      int64_t numElements = targetStrides[rank - 3];
      Value numElementsVal =
          rewriter.create<arith::ConstantIndexOp>(loc, numElements);
      auto shape = targetMemRefType.getShape();
      sizes[rank - 1] = rewriter.getIndexAttr(shape[rank - 1]);
      sizes[rank - 2] = rewriter.getIndexAttr(shape[rank - 2]);

      auto srcStride = rewriter.create<memref::SubViewOp>(loc, source, offsets,
                                                          sizes, strides);
      int64_t stride, width;
      assert(isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
             isStridedMultiDimMemrefType(targetMemRefType, stride, width));
      emitDMA(srcStride, target, zeroIndices, tileIndices, numElementsVal,
              getI32Const(loc, rewriter, stride),
              getI32Const(loc, rewriter, width));
    } else {
      int64_t numElements = sourceStrides[rank - 3];
      Value numElementsVal =
          rewriter.create<arith::ConstantIndexOp>(loc, numElements);
      auto shape = sourceMemRefType.getShape();
      sizes[rank - 1] = rewriter.getIndexAttr(shape[rank - 1]);
      sizes[rank - 2] = rewriter.getIndexAttr(shape[rank - 2]);

      auto dstStride = rewriter.create<memref::SubViewOp>(loc, target, offsets,
                                                          sizes, strides);
      int64_t stride, width;
      assert(isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
             isStridedMultiDimMemrefType(targetMemRefType, stride, width));
      emitDMA(source, dstStride, tileIndices, zeroIndices, numElementsVal,
              getI32Const(loc, rewriter, stride),
              getI32Const(loc, rewriter, width));
    }

    // Deallocate tag buffer
    rewriter.setInsertionPointAfter(outerLoop);
    rewriter.create<memref::DeallocOp>(loc, tagAlloc);
  } else {
    Value numElements =
        getI32Const(loc, rewriter, sourceMemRefType.getNumElements());
    if (isContiguousMemrefType(sourceMemRefType) &&
        isContiguousMemrefType(targetMemRefType)) {

      rewriter.create<memref::DmaStartOp>(loc, copyOp.getSource(), zeroIndices,
                                          copyOp.getTarget(), zeroIndices,
                                          numElements, tagAlloc, zeroIndex);
      rewriter.create<memref::DmaWaitOp>(loc, tagAlloc, tagIndex, numElements);
    } else {
      int64_t stride, width;
      assert(isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
             isStridedMultiDimMemrefType(targetMemRefType, stride, width));
      rewriter.create<memref::DmaStartOp>(
          loc, copyOp.getSource(), zeroIndices, copyOp.getTarget(), zeroIndices,
          numElements, tagAlloc, zeroIndex, getI32Const(loc, rewriter, stride),
          getI32Const(loc, rewriter, width));
      rewriter.create<memref::DmaWaitOp>(loc, tagAlloc, tagIndex, numElements);
    }

    // Deallocate tag buffer
    rewriter.create<memref::DeallocOp>(loc, tagAlloc);
  }

  // Erase the original copy operation
  rewriter.eraseOp(copyOp);
}

void HexmemCpyToDMAPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  CandidatesTy candidates;

  funcOp.walk([&](hexagonmem::CopyOp op) {
    populateCandidate(op, candidates);
    return WalkResult::advance();
  });

  for (auto op : candidates) {
    replaceWithDMA(op, funcOp, rewriter);
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexmemCpyToDMAPass() {
  return std::make_unique<HexmemCpyToDMAPass>();
}
