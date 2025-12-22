//===- UserDMADescriptors.h - Definitions of DMA Descriptors -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Reference: Qualcomm Hexagon User DMA Specification 80-V9418-29 Rev. D
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_USERDMA_DESCRIPTORS_H
#define HEXAGONBACKEND_BIN_RUNTIME_USERDMA_DESCRIPTORS_H

namespace hexagon {
namespace userdma {

// NOTE: Using 2D descriptor size even for 1D descriptors
#define DMA_DESC_2D_SIZE 32

// DMA State
// desc[0][3:0]
#define DESC_STATE_MASK 0x0000000F
#define DESC_STATE_SHIFT 0
#define DESC_STATE_READY 0

// desc[0][31:4]
// Descriptors addresses must be (minimum) 16 byte aligned
// -> Lower 4 bits masked to clear DMA Status
// -> But, descriptor address is not shifted
#define DESC_NEXT_MASK 0xFFFFFFF0
#define DESC_NEXT_SHIFT 0

// desc[1][23:0]
#define DESC_LENGTH_MASK 0x00FFFFFF
#define DESC_LENGTH_SHIFT 0

// desc[1][25:24]
#define DESC_DESCTYPE_MASK 0x03000000
#define DESC_DESCTYPE_SHIFT 24
#define DESC_DESCTYPE_1D 0
#define DESC_DESCTYPE_2D 1

// TODO: Definition?  Not in the spec.
// desc[1][26]
#define DESC_DSTCOMP_MASK 0x04000000
#define DESC_DSTCOMP_SHIFT 26
// desc[1][27]
#define DESC_SRCCOMP_MASK 0x08000000
#define DESC_SRCCOMP_SHIFT 27
#define DESC_COMP_NONE 0
#define DESC_COMP_DLBC 1

// desc[1][28]
#define DESC_BYPASSDST_MASK 0x10000000
#define DESC_BYPASSDST_SHIFT 28
// desc[1][29]
#define DESC_BYPASSSRC_MASK 0x20000000
#define DESC_BYPASSSRC_SHIFT 29
#define DESC_BYPASS_OFF 0
#define DESC_BYPASS_ON 1

// desc[1][30]
#define DESC_ORDER_MASK 0x40000000
#define DESC_ORDER_SHIFT 30
#define DESC_ORDER_NOORDER 0
#define DESC_ORDER_ORDER 1

// desc[1][31]
#define DESC_DONE_MASK 0x80000000
#define DESC_DONE_SHIFT 31
#define DESC_DONE_INCOMPLETE 0
#define DESC_DONE_COMPLETE 1

// desc[2]
#define DESC_SRC_MASK 0xFFFFFFFF
#define DESC_SRC_SHIFT 0

// desc[3]
#define DESC_DST_MASK 0xFFFFFFFF
#define DESC_DST_SHIFT 0

// desc[4][25:24]
#define DESC_CACHEALLOC_MASK 0x03000000
#define DESC_CACHEALLOC_SHIFT 24
#define DESC_CACHEALLOC_NONE 0
#define DESC_CACHEALLOC_WRITEONLY 1
#define DESC_CACHEALLOC_READONLY 2
#define DESC_CACHEALLOC_READWRITE 3

// TODO: Definition?  Not in the spec.
// desc[4][31:28]
#define DESC_PADDING_MASK 0xF0000000
#define DESC_PADDING_SHIFT 28

// desc[5][15:0]
#define DESC_ROIWIDTH_MASK 0x0000FFFF
#define DESC_ROIWIDTH_SHIFT 0

// desc[5][31:16]
#define DESC_ROIHEIGHT_MASK 0xFFFF0000
#define DESC_ROIHEIGHT_SHIFT 16

// desc[6][15:0]
#define DESC_SRCSTRIDE_MASK 0x0000FFFF
#define DESC_SRCSTRIDE_SHIFT 0

// desc[6][31:16]
#define DESC_DSTSTRIDE_MASK 0xFFFF0000
#define DESC_DSTSTRIDE_SHIFT 16

// desc[7][15:0]
#define DESC_SRCWIDTHOFFSET_MASK 0x0000FFFF
#define DESC_SRCWIDTHOFFSET_SHIFT 0

// desc[7][31:16]
#define DESC_DSTWIDTHOFFSET_MASK 0xFFFF0000
#define DESC_DSTWIDTHOFFSET_SHIFT 16

#define DMA_NULL_PTR 0

/**************************/
/* 1D (linear) descriptor */
/**************************/
struct DMADesc1D {
  unsigned int nextState;
  unsigned int doneOrderBypassCompDescTypeLength;
  unsigned int src;
  unsigned int dst;
};

/***********************/
/* 2D (box) descriptor */
/***********************/
struct DMADesc2D {
  unsigned int nextState;
  unsigned int doneOrderBypassCompDescTypeLength;
  unsigned int src;
  unsigned int dst;
  unsigned int allocationPadding;
  unsigned int roiHeightROIWidth;
  unsigned int dstStrideSrcStride;
  unsigned int dstWidthOffsetSrcWidthOffset;
};

// desc[0][3:0]
inline void dmaDescSetState(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->nextState) &= ~DESC_STATE_MASK;
  (dmaDesc1DPtr->nextState) |= ((v << DESC_STATE_SHIFT) & DESC_STATE_MASK);
}

// desc[0][31:4]
inline void dmaDescSetNext(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->nextState) &= ~DESC_NEXT_MASK;
  (dmaDesc1DPtr->nextState) |= ((v << DESC_NEXT_SHIFT) & DESC_NEXT_MASK);
}

// desc[1][23:0]
inline void dmaDescSetLength(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_LENGTH_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_LENGTH_SHIFT) & DESC_LENGTH_MASK);
}

// desc[1][25:24]
inline void dmaDescSetDescType(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_DESCTYPE_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_DESCTYPE_SHIFT) & DESC_DESCTYPE_MASK);
}

// TODO: Definition?  Not in the spec.
// desc[1][26]
inline void dmaDescSetDstComp(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_DSTCOMP_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_DSTCOMP_SHIFT) & DESC_DSTCOMP_MASK);
}

// TODO: Definition?  Not in the spec.
// desc[1][27]
inline void dmaDescSetSrcComp(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_SRCCOMP_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_SRCCOMP_SHIFT) & DESC_SRCCOMP_MASK);
}

// desc[1][28]
inline void dmaDescSetBypassDst(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_BYPASSDST_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_BYPASSDST_SHIFT) & DESC_BYPASSDST_MASK);
}

// desc[1][29]
inline void dmaDescSetBypassSrc(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_BYPASSSRC_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_BYPASSSRC_SHIFT) & DESC_BYPASSSRC_MASK);
}

// desc[1][30]
inline void dmaDescSetOrder(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_ORDER_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_ORDER_SHIFT) & DESC_ORDER_MASK);
}

// desc[1][31]
inline void dmaDescSetDone(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) &= ~DESC_DONE_MASK;
  (dmaDesc1DPtr->doneOrderBypassCompDescTypeLength) |=
      ((v << DESC_DONE_SHIFT) & DESC_DONE_MASK);
}

// desc[1][31]
inline unsigned int dmaDescGetDone(void *dmaDescPtr) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  volatile unsigned int *doneAsVolatile = static_cast<volatile unsigned int *>(
      &(dmaDesc1DPtr->doneOrderBypassCompDescTypeLength));
  // Descriptor can be modified by DMA engine as well, make sure we see the
  // updated value.
  // The volatile read prevents compiler optimizations that treat the read
  // as an invariant, or move side-effect ops across the volatile read.
  unsigned int doneVal = *doneAsVolatile;
  return ((doneVal & DESC_DONE_MASK) >> DESC_DONE_SHIFT);
}

// desc[2]
inline void dmaDescSetSrc(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->src) &= ~DESC_SRC_MASK;
  (dmaDesc1DPtr->src) |= ((v << DESC_SRC_SHIFT) & DESC_SRC_MASK);
}

// desc[3]
inline void dmaDescSetDst(void *dmaDescPtr, unsigned int v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->dst) &= ~DESC_DST_MASK;
  (dmaDesc1DPtr->dst) |= ((v << DESC_DST_SHIFT) & DESC_DST_MASK);
}

// desc[4][25:24]
inline void dmaDescSetCacheAlloc(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->allocationPadding) &= ~DESC_CACHEALLOC_MASK;
  (dmaDesc2DPtr->allocationPadding) |=
      ((v << DESC_CACHEALLOC_SHIFT) & DESC_CACHEALLOC_MASK);
}

// TODO: Definition?  Not in the spec.
// desc[4][31:28]
inline void dmaDescSetPadding(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->allocationPadding) &= ~DESC_PADDING_MASK;
  (dmaDesc2DPtr->allocationPadding) |=
      ((v << DESC_PADDING_SHIFT) & DESC_PADDING_MASK);
}

// desc[5][15:0]
inline void dmaDescSetROIWidth(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->roiHeightROIWidth) &= ~DESC_ROIWIDTH_MASK;
  (dmaDesc2DPtr->roiHeightROIWidth) |=
      ((v << DESC_ROIWIDTH_SHIFT) & DESC_ROIWIDTH_MASK);
}

// desc[5][31:16]
inline void dmaDescSetROIHeight(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->roiHeightROIWidth) &= ~DESC_ROIHEIGHT_MASK;
  (dmaDesc2DPtr->roiHeightROIWidth) |=
      ((v << DESC_ROIHEIGHT_SHIFT) & DESC_ROIHEIGHT_MASK);
}

// desc[6][15:0]
inline void dmaDescSetSrcStride(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->dstStrideSrcStride) &= ~DESC_SRCSTRIDE_MASK;
  (dmaDesc2DPtr->dstStrideSrcStride) |=
      ((v << DESC_SRCSTRIDE_SHIFT) & DESC_SRCSTRIDE_MASK);
}

// desc[6][31:16]
inline void dmaDescSetDstStride(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->dstStrideSrcStride) &= ~DESC_DSTSTRIDE_MASK;
  (dmaDesc2DPtr->dstStrideSrcStride) |=
      ((v << DESC_DSTSTRIDE_SHIFT) & DESC_DSTSTRIDE_MASK);
}

// desc[7][15:0]
inline void dmaDescSetSrcWidthOffset(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->dstWidthOffsetSrcWidthOffset) &= ~DESC_SRCWIDTHOFFSET_MASK;
  (dmaDesc2DPtr->dstWidthOffsetSrcWidthOffset) |=
      ((v << DESC_SRCWIDTHOFFSET_SHIFT) & DESC_SRCWIDTHOFFSET_MASK);
}

// desc[7][31:16]
inline void dmaDescSetDstWidthOffset(void *dmaDescPtr, unsigned int v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->dstWidthOffsetSrcWidthOffset) &= ~DESC_DSTWIDTHOFFSET_MASK;
  (dmaDesc2DPtr->dstWidthOffsetSrcWidthOffset) |=
      ((v << DESC_DSTWIDTHOFFSET_SHIFT) & DESC_DSTWIDTHOFFSET_MASK);
}
} // namespace userdma
} // namespace hexagon

#endif // HEXAGONBACKEND_BIN_RUNTIME_USERDMA_DESCRIPTORS_H
