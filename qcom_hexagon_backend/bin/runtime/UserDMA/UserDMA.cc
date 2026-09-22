//===- UserDMA.cc - Implementation of UserDMA API -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Calls to member functions are wrapped inside user-facing C style DMA APIs.
//
//===----------------------------------------------------------------------===//

#include "UserDMA.h"
#include <cassert>
#include <cstdint>
#include <cstring>

namespace hexagon {
namespace userdma {

namespace {
// Hand back a token whose wait() returns even though no transfer was started.
// DMAToLLVMPass stores the token and never inspects `status`, so a rejected
// transfer that returned 0 could make a later dma_wait(0) spin on an unrelated
// descriptor. The descriptor below is left unlinked (the DMA engine never sees
// it) and marked done, so the caller's wait() completes immediately.
uint32_t enqueueRejectedDesc(RingBuffer<DMADesc2D> *queue, DMAStatus *status) {
  uint32_t token = 0;
  DMADesc2D *dmaDesc = queue->alloc(token);
  std::memset(dmaDesc, 0, sizeof(DMADesc2D));
  dmaDescSetState(dmaDesc, DESC_STATE_READY);
  dmaDescSetNext(dmaDesc, DMA_NULL_PTR);
  dmaDescSetDone(dmaDesc, DESC_DONE_COMPLETE);
  *status = DMAFailure;
  return token;
}
} // namespace

bool inFlight(void *ptr) {
  DMADesc2D *dmaDesc = static_cast<DMADesc2D *>(ptr);
  dmpoll(); // Catch any exception occured during DMA transfer
  unsigned int done = dmaDescGetDone(dmaDesc);
  return (done != DESC_DONE_COMPLETE);
};

unsigned int UserDMA::init() { return dmpause() & DM0_STATUS_MASK; }

uint32_t UserDMA::copy(void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS,
                       uint32_t numBytes, bool bypassCacheSrc,
                       bool bypassCacheDst, DMAStatus *status) {
  // length limited to 24 bits
  if (numBytes > DESC_LENGTH_MASK) {
    *status = DMAFailure;
    return 0;
  }

  // source address limited to 32 bits
  uint64_t src64 = reinterpret_cast<uint64_t>(src);
  if (!src64 || src64 > DESC_SRC_MASK) {
    *status = DMAFailure;
    return 0;
  }

  // destination address limited to 32 bits
  uint64_t dst64 = reinterpret_cast<uint64_t>(dst);
  if (!dst64 || dst64 > DESC_DST_MASK) {
    *status = DMAFailure;
    return 0;
  }

  uint32_t src32 = static_cast<uint32_t>(src64);
  uint32_t dst32 = static_cast<uint32_t>(dst64);

  // get pointer to next descriptor
  DMADesc2D *dmaDesc;
  uint32_t token;

  dmaDesc = dmaQueue->alloc(token);

  // A slot reused after a 2D transfer may still carry desc_type=9 in word4.
  // Clear it; a 1D descriptor leaves word4 reserved.
  dmaDesc->descType = 0;

  // populate descriptor fields
  dmaDescSetState(dmaDesc, DESC_STATE_READY);
  dmaDescSetDone(dmaDesc, DESC_DONE_INCOMPLETE);
  dmaDescSetNext(dmaDesc, DMA_NULL_PTR);
  dmaDescSetLength(dmaDesc, numBytes);
  dmaDescSetDescSize(dmaDesc, DESC_DESCSIZE_1D);
  dmaDescSetDstComp(dmaDesc, DESC_COMP_NONE);
  dmaDescSetSrcComp(dmaDesc, DESC_COMP_NONE);

  // For now, disable bypass cache
  dmaDescSetBypassDst(dmaDesc,
                      bypassCacheDst ? DESC_BYPASS_ON : DESC_BYPASS_OFF);
  dmaDescSetBypassSrc(dmaDesc,
                      bypassCacheSrc ? DESC_BYPASS_ON : DESC_BYPASS_OFF);

  dmaDescSetOrder(dmaDesc, DESC_ORDER_NOORDER);
  dmaDescSetDone(dmaDesc, DESC_DONE_INCOMPLETE);
  dmaDescSetSrc(dmaDesc, src32);
  dmaDescSetDst(dmaDesc, dst32);

  if (isFirstDMA) {
    // `dmstart` first descriptor
    dmstart(dmaDesc);
    isFirstDMA = false;
  } else {
    // `dmlink` descriptor to tail descriptor
    dmlink(tailDMADesc, dmaDesc);
  }

  // update tail
  tailDMADesc = dmaDesc;
  *status = DMASuccess;
  return token;
}

uint32_t UserDMA::copy2D(void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS,
                         uint32_t width, uint32_t height, uint32_t srcStride,
                         uint32_t dstStride, bool bypassCacheSrc,
                         bool bypassCacheDst, bool isOrdered,
                         uint32_t cacheAllocationPolicy, DMAStatus *status) {

  *status = DMAFailure;

  // The v75+ descriptor holds width/row_size and the strides in 24-bit fields
  // and height in 16 bits. The setters only mask, so an out-of-range geometry
  // would be truncated silently and the engine would copy the wrong rows (on
  // v79 a src_stride of 65600 was lowered as 64). Reject instead. This is a
  // refusal, not an assert: the device runtime is built at -O2 without
  // -DNDEBUG, so assert() would abort the DSP on a data-dependent value.
  if (!dma2DGeometryFits(width, height, srcStride, dstStride)) {
    return enqueueRejectedDesc(dmaQueue, status);
  }

  // source address limited to 32 bits
  uint64_t src64 = reinterpret_cast<uint64_t>(src);
  if (!src64 || src64 > DESC_SRC_MASK) {
    return enqueueRejectedDesc(dmaQueue, status);
  }

  // destination address limited to 32 bits
  uint64_t dst64 = reinterpret_cast<uint64_t>(dst);
  if (!dst64 || dst64 > DESC_DST_MASK) {
    return enqueueRejectedDesc(dmaQueue, status);
  }

  uint32_t src32 = static_cast<uint32_t>(src64);
  uint32_t dst32 = static_cast<uint32_t>(dst64);

  // get pointer to next descriptor
  DMADesc2D *dmaDesc;
  uint32_t token;

  dmaDesc = dmaQueue->alloc(token);
  // This transfer is not ordered wrt earlier transfers;
  // DMA Done bit cleared; No read/write cache allocation policy
  std::memset(dmaDesc, 0, sizeof(DMADesc2D));

  // populate descriptor fields
  dmaDescSetState(dmaDesc, DESC_STATE_READY);
  dmaDescSetDone(dmaDesc, DESC_DONE_INCOMPLETE);
  dmaDescSetNext(dmaDesc, DMA_NULL_PTR);
  dmaDescSetDescSize(dmaDesc, DESC_DESCSIZE_2D);
  dmaDescSetDescType(dmaDesc, DESC_TYPE_2D_24BIT);
  dmaDescSetDstComp(dmaDesc, DESC_COMP_NONE);
  dmaDescSetSrcComp(dmaDesc, DESC_COMP_NONE);

  dmaDescSetRowSize(dmaDesc, width);
  dmaDescSetNrows(dmaDesc, height);
  dmaDescSetSrcStride(dmaDesc, srcStride);
  dmaDescSetDstStride(dmaDesc, dstStride);
  dmaDescSetOffset(dmaDesc, 0);

  dmaDescSetBypassDst(dmaDesc,
                      bypassCacheDst ? DESC_BYPASS_ON : DESC_BYPASS_OFF);
  dmaDescSetBypassSrc(dmaDesc,
                      bypassCacheSrc ? DESC_BYPASS_ON : DESC_BYPASS_OFF);

  dmaDescSetOrder(dmaDesc, DESC_ORDER_NOORDER);
  dmaDescSetDone(dmaDesc, DESC_DONE_INCOMPLETE);
  dmaDescSetSrc(dmaDesc, src32);
  dmaDescSetDst(dmaDesc, dst32);

  // On first call use dmstart instruction and reset isFirstDMA flag.
  // On subsequent calls use dmlink instruction, linking previous(tailDMADesc)
  // to current.
  if (isFirstDMA) {
    // `dmstart` first descriptor
    dmstart(dmaDesc);
    isFirstDMA = false;
  } else {
    // `dmlink` descriptor to tail descriptor
    dmlink(tailDMADesc, dmaDesc);
  }

  // update tail
  tailDMADesc = dmaDesc;
  *status = DMASuccess;
  return token;
}

void UserDMA::wait(uint32_t token) {
  DMADesc2D *dmaDesc = dmaQueue->getDataPtr(token);
  assert(dmaDesc != nullptr);
  while (inFlight(dmaDesc))
    ; // keep looping till DMA transfer is in progress
}

UserDMA::UserDMA(int fifoLen) {
  // reset DMA engine
  unsigned int status = init();
  assert(status == DM0_STATUS_IDLE);

  dmaQueue = new RingBuffer<DMADesc2D>(fifoLen, inFlight);
}

UserDMA::~UserDMA() {
  init(); // stop DMA engine
  delete dmaQueue;
}
} // namespace userdma
} // namespace hexagon
