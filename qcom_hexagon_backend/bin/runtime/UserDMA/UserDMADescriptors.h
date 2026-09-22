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

#include <cstdint>

namespace hexagon {
namespace userdma {

// NOTE: Using 2D descriptor size even for 1D descriptors
#define DMA_DESC_2D_SIZE 32

// The 2D descriptor below is the v75+ 24-bit layout, not the pre-v75 16-bit one.
// The 16-bit layout split row/strides into 16-bit halves and selected the mode
// through word1[25:24] ("desc_type" 0/1). The v75+ layout is wider and selects
// the mode with word4[7:0] = 9:
//
//   word1[23:0]  dst_stride
//   word1[25:24] desc_size   (0 = 1D, 1 = 2D)
//   word4[7:0]   desc_type   (9 for the 24-bit 2D layout)
//   word5[23:0]  row_size
//   word5[31:24] nrows_lo
//   word6[7:0]   nrows_hi
//   word6[31:8]  src_stride
//   word7[23:0]  offset
//
// The 16-bit layout silently truncated strides to 16 bits: on v79 a src_stride
// of 65600 was lowered as 64 and the engine copied the wrong rows with no error
// (logs/dma2d_probe/REPORT.txt). There is deliberately no 16-bit fallback here;
// this runtime targets v75+ (v79 in this repo). If this header is ever built for
// a pre-v75 target the layout must be restored behind a version check.
//
// Reference implementation: llama.cpp htp/dma-queue.h dma_descriptor_2d.

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

// desc[1][23:0]: 1D transfer length, or 2D destination stride.
#define DESC_LENGTH_MASK 0x00FFFFFF
#define DESC_LENGTH_SHIFT 0
#define DESC_DSTSTRIDE_MASK 0x00FFFFFF
#define DESC_DSTSTRIDE_SHIFT 0

// desc[1][25:24]: descriptor size (0 = 1D, 1 = 2D), not the v75+ mode selector.
#define DESC_DESCSIZE_MASK 0x03000000
#define DESC_DESCSIZE_SHIFT 24
#define DESC_DESCSIZE_1D 0
#define DESC_DESCSIZE_2D 1

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

// desc[4][7:0]: 24-bit 2D descriptor type. 9 selects the v75+ layout.
#define DESC_TYPE_MASK 0x000000FF
#define DESC_TYPE_SHIFT 0
#define DESC_TYPE_2D_24BIT 9

// desc[5][23:0]
#define DESC_ROWSIZE_MASK 0x00FFFFFF
#define DESC_ROWSIZE_SHIFT 0

// desc[5][31:24]
#define DESC_NROWSLO_MASK 0xFF000000
#define DESC_NROWSLO_SHIFT 24

// desc[6][7:0]
#define DESC_NROWSHI_MASK 0x000000FF
#define DESC_NROWSHI_SHIFT 0

// desc[6][31:8]
#define DESC_SRCSTRIDE_MASK 0xFFFFFF00
#define DESC_SRCSTRIDE_SHIFT 8

// desc[7][23:0]
#define DESC_OFFSET_MASK 0x00FFFFFF
#define DESC_OFFSET_SHIFT 0

// Largest value each 2D geometry field can hold. nrows spans word5[31:24] and
// word6[7:0], i.e. 16 bits total; the rest are 24 bits.
#define DESC_STRIDE_MAX 0x00FFFFFF
#define DESC_ROWSIZE_MAX 0x00FFFFFF
#define DESC_NROWS_MAX 0x0000FFFF

#define DMA_NULL_PTR 0

/**************************/
/* 1D (linear) descriptor */
/**************************/
struct DMADesc1D {
  uint32_t nextState;
  uint32_t lengthDescSizeDoneOrderBypassComp;
  uint32_t src;
  uint32_t dst;
};

/***********************/
/* 2D (box) descriptor */
/***********************/
struct DMADesc2D {
  uint32_t nextState;
  uint32_t dstStrideDescSizeDoneOrderBypassComp;
  uint32_t src;
  uint32_t dst;
  uint32_t descType;
  uint32_t rowSizeNrowsLo;
  uint32_t nrowsHiSrcStride;
  uint32_t offset;
};

static_assert(sizeof(DMADesc2D) == DMA_DESC_2D_SIZE,
              "2D DMA descriptor must be 32 bytes");

// desc[0][3:0]
inline void dmaDescSetState(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->nextState) &= ~DESC_STATE_MASK;
  (dmaDesc1DPtr->nextState) |= ((v << DESC_STATE_SHIFT) & DESC_STATE_MASK);
}

// desc[0][31:4]
inline void dmaDescSetNext(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->nextState) &= ~DESC_NEXT_MASK;
  (dmaDesc1DPtr->nextState) |= ((v << DESC_NEXT_SHIFT) & DESC_NEXT_MASK);
}

// desc[1][23:0]
inline void dmaDescSetLength(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_LENGTH_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_LENGTH_SHIFT) & DESC_LENGTH_MASK);
}

// desc[1][25:24]. Formerly (mis)named dmaDescSetDescType; these bits are
// desc_size, while the v75+ mode selector is dmaDescSetDescType below.
inline void dmaDescSetDescSize(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_DESCSIZE_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_DESCSIZE_SHIFT) & DESC_DESCSIZE_MASK);
}

// TODO: Definition?  Not in the spec.
// desc[1][26]
inline void dmaDescSetDstComp(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_DSTCOMP_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_DSTCOMP_SHIFT) & DESC_DSTCOMP_MASK);
}

// TODO: Definition?  Not in the spec.
// desc[1][27]
inline void dmaDescSetSrcComp(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_SRCCOMP_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_SRCCOMP_SHIFT) & DESC_SRCCOMP_MASK);
}

// desc[1][28]
inline void dmaDescSetBypassDst(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_BYPASSDST_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_BYPASSDST_SHIFT) & DESC_BYPASSDST_MASK);
}

// desc[1][29]
inline void dmaDescSetBypassSrc(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_BYPASSSRC_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_BYPASSSRC_SHIFT) & DESC_BYPASSSRC_MASK);
}

// desc[1][30]
inline void dmaDescSetOrder(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_ORDER_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_ORDER_SHIFT) & DESC_ORDER_MASK);
}

// desc[1][31]
inline void dmaDescSetDone(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) &= ~DESC_DONE_MASK;
  (dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp) |=
      ((v << DESC_DONE_SHIFT) & DESC_DONE_MASK);
}

// desc[1][31]
inline uint32_t dmaDescGetDone(void *dmaDescPtr) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  volatile uint32_t *doneAsVolatile = static_cast<volatile uint32_t *>(
      &(dmaDesc1DPtr->lengthDescSizeDoneOrderBypassComp));
  // Descriptor can be modified by DMA engine as well, make sure we see the
  // updated value.
  // The volatile read prevents compiler optimizations that treat the read
  // as an invariant, or move side-effect ops across the volatile read.
  uint32_t doneVal = *doneAsVolatile;
  return ((doneVal & DESC_DONE_MASK) >> DESC_DONE_SHIFT);
}

// desc[2]
inline void dmaDescSetSrc(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->src) &= ~DESC_SRC_MASK;
  (dmaDesc1DPtr->src) |= ((v << DESC_SRC_SHIFT) & DESC_SRC_MASK);
}

// desc[3]
inline void dmaDescSetDst(void *dmaDescPtr, uint32_t v) {
  DMADesc1D *dmaDesc1DPtr = reinterpret_cast<DMADesc1D *>(dmaDescPtr);
  (dmaDesc1DPtr->dst) &= ~DESC_DST_MASK;
  (dmaDesc1DPtr->dst) |= ((v << DESC_DST_SHIFT) & DESC_DST_MASK);
}

// desc[1][23:0]: 2D destination stride. Shares word1 with the 1D length field.
inline void dmaDescSetDstStride(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->dstStrideDescSizeDoneOrderBypassComp) &= ~DESC_DSTSTRIDE_MASK;
  (dmaDesc2DPtr->dstStrideDescSizeDoneOrderBypassComp) |=
      ((v << DESC_DSTSTRIDE_SHIFT) & DESC_DSTSTRIDE_MASK);
}

// desc[4][7:0]: 24-bit 2D mode selector (9).
inline void dmaDescSetDescType(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->descType) &= ~DESC_TYPE_MASK;
  (dmaDesc2DPtr->descType) |= ((v << DESC_TYPE_SHIFT) & DESC_TYPE_MASK);
}

// desc[5][23:0]
inline void dmaDescSetRowSize(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->rowSizeNrowsLo) &= ~DESC_ROWSIZE_MASK;
  (dmaDesc2DPtr->rowSizeNrowsLo) |=
      ((v << DESC_ROWSIZE_SHIFT) & DESC_ROWSIZE_MASK);
}

// desc[5][31:24] (low 8 bits) + desc[6][7:0] (high 8 bits)
inline void dmaDescSetNrows(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->rowSizeNrowsLo) &= ~DESC_NROWSLO_MASK;
  (dmaDesc2DPtr->rowSizeNrowsLo) |=
      ((v << DESC_NROWSLO_SHIFT) & DESC_NROWSLO_MASK);
  (dmaDesc2DPtr->nrowsHiSrcStride) &= ~DESC_NROWSHI_MASK;
  (dmaDesc2DPtr->nrowsHiSrcStride) |= ((v >> 8) & DESC_NROWSHI_MASK);
}

// desc[6][31:8]
inline void dmaDescSetSrcStride(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->nrowsHiSrcStride) &= ~DESC_SRCSTRIDE_MASK;
  (dmaDesc2DPtr->nrowsHiSrcStride) |=
      ((v << DESC_SRCSTRIDE_SHIFT) & DESC_SRCSTRIDE_MASK);
}

// desc[7][23:0]
inline void dmaDescSetOffset(void *dmaDescPtr, uint32_t v) {
  DMADesc2D *dmaDesc2DPtr = reinterpret_cast<DMADesc2D *>(dmaDescPtr);
  (dmaDesc2DPtr->offset) &= ~DESC_OFFSET_MASK;
  (dmaDesc2DPtr->offset) |= ((v << DESC_OFFSET_SHIFT) & DESC_OFFSET_MASK);
}

// True if every 2D geometry field fits its descriptor bit width. Callers must
// check this before programming the setters: the setters only mask, so an
// out-of-range value would be truncated silently.
inline bool dma2DGeometryFits(uint32_t width, uint32_t height,
                              uint32_t srcStride, uint32_t dstStride) {
  return width <= DESC_ROWSIZE_MAX && height <= DESC_NROWS_MAX &&
         srcStride <= DESC_STRIDE_MAX && dstStride <= DESC_STRIDE_MAX;
}

} // namespace userdma
} // namespace hexagon

#endif // HEXAGONBACKEND_BIN_RUNTIME_USERDMA_DESCRIPTORS_H
