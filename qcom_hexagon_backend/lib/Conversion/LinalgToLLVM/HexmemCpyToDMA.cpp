//===- HexmemCpyToDMA.cpp - hexagonmem.copy to memref.dma* conversion ====-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting of hexagonmem.copy and memref.copy ops to
// memref.dma* ops
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Common/Utils.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexmem-cpy-to-dma"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXMEMCPYTODMA
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct HexmemCpyToDMAPass
    : public ::impl::HexmemCpyToDMABase<HexmemCpyToDMAPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override;
};

static Value getI32Const(Location loc, IRRewriter &rewriter, int val) {
  return arith::ConstantIndexOp::create(rewriter, loc, val);
}

// Trace a memref value back through view-like ops (subview/view/cast/
// reinterpret_cast) to its defining op and report whether it is a function-
// local DDR allocation (memref.alloc / memref.alloca in the default address
// space).
//
// Such DDR buffers are written on the DSP (e.g. linalg.fill + memref.copy used
// to materialize masked loads, or reduction scalar temporaries) and live in
// the DSP cache. The UDMA engine reads/writes DDR directly and is not coherent
// with that cache, so converting a copy whose DDR endpoint is one of these
// freshly written local allocations into a DMA reads/writes stale DDR (observed
// as all-zero reduction inputs/outputs for masked tl.sum / rms_norm). Leaving
// these as memref.copy keeps them as a coherent DSP-side copy.
//
// Local allocations in VTCM are fine: VTCM is the UDMA engine's own fast memory
// and is the intended DMA endpoint, so only DDR (default address space) local
// allocations are flagged here.
static bool isLocalDDRAllocation(mlir::Value memref) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!memrefType)
    return false;
  int memSpace;
  if (!isMemorySpaceIntTypeOrDefault(memrefType, memSpace) ||
      memSpace != hexagon::DEFAULT_DDR_ADDRESS_SPACE)
    return false;

  mlir::Value cur = memref;
  while (cur) {
    mlir::Operation *def = cur.getDefiningOp();
    if (!def)
      return false; // block argument (e.g. function argument) - not local
    if (mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp>(def))
      return true;
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def)) {
      cur = view.getViewSource();
      continue;
    }
    return false;
  }
  return false;
}

template <typename CopyOpTy> static bool isValidCandidate(CopyOpTy op) {
  auto sourceType = op.getSource().getType();
  auto targetType = op.getTarget().getType();

  if (isa<crouton::CroutonType>(sourceType) ||
      isa<crouton::CroutonType>(targetType))
    return false;

  auto sourceMemRefType = dyn_cast<MemRefType>(sourceType);
  auto targetMemRefType = dyn_cast<MemRefType>(targetType);

  assert(sourceMemRefType && targetMemRefType &&
         "Expected Memref type as source and target to copy op");

  // Skip lowering for non-static
  if (!sourceMemRefType.hasStaticShape() || !targetMemRefType.hasStaticShape())
    return false;

  // Do not lower copies whose DDR endpoint is a function-local allocation.
  // These buffers are written on the DSP and are not cache-coherent with the
  // UDMA engine, so a DMA would read/write stale DDR. See
  // isLocalDDRAllocation(). This covers masked-load staging buffers
  // (linalg.fill + partial copy) and reduction scalar temporaries, both of
  // which otherwise produced all-zero results (tl.sum / softmax / rms_norm).
  if (isLocalDDRAllocation(op.getSource()) ||
      isLocalDDRAllocation(op.getTarget()))
    return false;

  int sourceMemSpace, targetMemSpace;
  // Unknown memory space - don't lower
  if (!isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace) ||
      !isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace))
    return false;

  if (targetMemSpace == sourceMemSpace)
    return false;

  // If any of source/target is not (multiDim-strided or contiguous), don't
  // lower
  auto isContiguousOrMultiDimMemrefType = [&](MemRefType type) {
    int64_t stride, width;
    return isContiguousMemrefType(type) ||
           isStridedMultiDimMemrefType(type, stride, width);
  };

  if (!isContiguousOrMultiDimMemrefType(sourceMemRefType) ||
      !isContiguousOrMultiDimMemrefType(targetMemRefType))
    return false;

  // Skip DMA conversion for strided multi-dim memrefs unless
  //  transferring DDR->VTCM (source) or VTCM->DDR (target)
  int64_t stride, width;
  if ((isStridedMultiDimMemrefType(sourceMemRefType, stride, width) &&
       !(sourceMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
         targetMemSpace == VTCM_ADDRESS_SPACE)) ||
      (isStridedMultiDimMemrefType(targetMemRefType, stride, width) &&
       !(targetMemSpace == DEFAULT_DDR_ADDRESS_SPACE &&
         sourceMemSpace == VTCM_ADDRESS_SPACE)))
    return false;

  return true;
}

template <typename CopyOpTy>
void replaceWithDMA(CopyOpTy copyOp, FunctionOpInterface funcOp,
                    IRRewriter &rewriter, Value zero, Value one) {
  rewriter.setInsertionPoint(copyOp.getOperation());
  auto loc = copyOp->getLoc();
  auto source = copyOp.getSource();
  auto target = copyOp.getTarget();

  auto sourceType = source.getType();
  auto targetType = target.getType();
  assert(isa<MemRefType>(sourceType) && isa<MemRefType>(targetType));

  MemRefType sourceMemRefType = cast<MemRefType>(sourceType);
  MemRefType targetMemRefType = cast<MemRefType>(targetType);
  int64_t rank = sourceMemRefType.getRank();

  // Extract strides and offsets for source and target memrefs
  SmallVector<int64_t, 4> sourceStrides, targetStrides;
  int64_t srcOffset, targetOffset;

  if (failed(sourceMemRefType.getStridesAndOffset(sourceStrides, srcOffset)) ||
      failed(targetMemRefType.getStridesAndOffset(targetStrides, targetOffset)))
    return;

  int sourceMemSpace, targetMemSpace;
  // Unknown memory space - don't lower
  if (!(isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace)) ||
      !(isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace)))
    return;

  // All early returns above are before any IR modification.
  // Allocate a tag buffer for DMA operations
  auto tagType = MemRefType::get({1}, rewriter.getI32Type());
  Value tagAlloc = memref::AllocOp::create(rewriter, loc << "dma_tag", tagType);

  // Reuse shared constants for tag and index access
  SmallVector<Value, 1> tagIndex = {zero};

  SmallVector<Value, 8> zeroIndices(rank, zero);
  SmallVector<Value, 8> tileIndices(rank, zero);
  SmallVector<Value, 2> zeroIndex(1, zero);

  auto emitDMA = [loc, &rewriter, &tagAlloc, &tagIndex](
                     Value src, Value dst, SmallVector<Value, 8> srcIndices,
                     SmallVector<Value, 8> destIndices, Value numElements,
                     Value stride, Value width) {
    memref::DmaStartOp::create(rewriter, loc, src, srcIndices, dst, destIndices,
                               numElements, tagAlloc, tagIndex, stride, width);
    memref::DmaWaitOp::create(rewriter, loc, tagAlloc, tagIndex, numElements);
  };

  if ((rank > 2) && (!isContiguousMemrefType(sourceMemRefType) ||
                     !isContiguousMemrefType(targetMemRefType))) {
    // Generate nested loops for dimensions beyond the last two
    SmallVector<Value, 8> loopIvs;
    scf::ForOp outerLoop;
    scf::ForOp curLoop;

    // Create inner loops for the remaining (rank - 2) dimensions
    for (int64_t i = 0; i < rank - 2; ++i) {
      curLoop = scf::ForOp::create(
          rewriter, loc, zero,
          getI32Const(loc, rewriter, sourceMemRefType.getDimSize(i)), one);
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
      Value numElementsVal = arith::ConstantIndexOp::create(
          rewriter, loc << "dma_num_elements", numElements);
      auto shape = targetMemRefType.getShape();
      sizes[rank - 1] = rewriter.getIndexAttr(shape[rank - 1]);
      sizes[rank - 2] = rewriter.getIndexAttr(shape[rank - 2]);

      auto srcStride = memref::SubViewOp::create(rewriter, loc, source, offsets,
                                                 sizes, strides);
      int64_t stride = 0, width = 0;
      bool found =
          isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
          isStridedMultiDimMemrefType(targetMemRefType, stride, width);
      assert(found && "Expected at least one strided multi-dim memref");
      emitDMA(srcStride, target, zeroIndices, tileIndices, numElementsVal,
              getI32Const(loc, rewriter, stride),
              getI32Const(loc, rewriter, width));
    } else {
      int64_t numElements = sourceStrides[rank - 3];
      Value numElementsVal = arith::ConstantIndexOp::create(
          rewriter, loc << "dma_num_elements", numElements);
      auto shape = sourceMemRefType.getShape();
      sizes[rank - 1] = rewriter.getIndexAttr(shape[rank - 1]);
      sizes[rank - 2] = rewriter.getIndexAttr(shape[rank - 2]);

      auto dstStride = memref::SubViewOp::create(rewriter, loc, target, offsets,
                                                 sizes, strides);
      int64_t stride = 0, width = 0;
      bool found =
          isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
          isStridedMultiDimMemrefType(targetMemRefType, stride, width);
      assert(found && "Expected at least one strided multi-dim memref");
      emitDMA(source, dstStride, tileIndices, zeroIndices, numElementsVal,
              getI32Const(loc, rewriter, stride),
              getI32Const(loc, rewriter, width));
    }

    // Deallocate tag buffer
    rewriter.setInsertionPointAfter(outerLoop);
    memref::DeallocOp::create(rewriter, loc, tagAlloc);
  } else {
    Value numElements = getI32Const(loc << "dma_num_elements", rewriter,
                                    sourceMemRefType.getNumElements());
    if (isContiguousMemrefType(sourceMemRefType) &&
        isContiguousMemrefType(targetMemRefType)) {

      memref::DmaStartOp::create(rewriter, loc, copyOp.getSource(), zeroIndices,
                                 copyOp.getTarget(), zeroIndices, numElements,
                                 tagAlloc, zeroIndex);
      memref::DmaWaitOp::create(rewriter, loc, tagAlloc, tagIndex, numElements);
    } else {
      int64_t stride = 0, width = 0;
      bool found =
          isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
          isStridedMultiDimMemrefType(targetMemRefType, stride, width);
      assert(found && "Expected at least one strided multi-dim memref");
      memref::DmaStartOp::create(rewriter, loc, copyOp.getSource(), zeroIndices,
                                 copyOp.getTarget(), zeroIndices, numElements,
                                 tagAlloc, zeroIndex,
                                 getI32Const(loc, rewriter, stride),
                                 getI32Const(loc, rewriter, width));
      memref::DmaWaitOp::create(rewriter, loc, tagAlloc, tagIndex, numElements);
    }

    // Deallocate tag buffer
    memref::DeallocOp::create(rewriter, loc, tagAlloc);
  }

  // Erase the original copy operation
  rewriter.eraseOp(copyOp);
}

void HexmemCpyToDMAPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  SmallVector<Operation *> candidates;

  funcOp.walk([&](hexagonmem::CopyOp op) {
    if (isValidCandidate(op)) {
      DBG("selected candidate: " << op);
      candidates.push_back(op.getOperation());
    }
    return WalkResult::advance();
  });

  funcOp.walk([&](memref::CopyOp op) {
    if (isValidCandidate(op)) {
      DBG("selected candidate: " << op);
      candidates.push_back(op.getOperation());
    }
    return WalkResult::advance();
  });

  if (candidates.empty())
    return;

  // Create shared constants at the function entry, reused across all DMA
  // replacements to avoid duplicate arith.constant ops.
  auto loc = funcOp.getLoc();
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

  for (auto *op : candidates) {
    if (auto hexOp = dyn_cast<hexagonmem::CopyOp>(op))
      replaceWithDMA(hexOp, funcOp, rewriter, zero, one);
    else if (auto memOp = dyn_cast<memref::CopyOp>(op))
      replaceWithDMA(memOp, funcOp, rewriter, zero, one);
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexmemCpyToDMAPass() {
  return std::make_unique<HexmemCpyToDMAPass>();
}
