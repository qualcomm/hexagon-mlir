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

namespace hexagon {
namespace userdma {
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

  // populate descriptor fields
  dmaDescSetState(dmaDesc, DESC_STATE_READY);
  dmaDescSetDone(dmaDesc, DESC_DONE_INCOMPLETE);
  dmaDescSetNext(dmaDesc, DMA_NULL_PTR);
  dmaDescSetLength(dmaDesc, numBytes);
  dmaDescSetDescType(dmaDesc, DESC_DESCTYPE_1D);
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

  // source address limited to 32 bits
  uint64_t src64 = reinterpret_cast<uint64_t>(src);
  if (!src64 || src64 > DESC_SRC_MASK) {
    return 0;
  }

  // destination address limited to 32 bits
  uint64_t dst64 = reinterpret_cast<uint64_t>(dst);
  if (!dst64 || dst64 > DESC_DST_MASK) {
    return 0;
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
  dmaDescSetDescType(dmaDesc, DESC_DESCTYPE_2D);
  dmaDescSetDstComp(dmaDesc, DESC_COMP_NONE);
  dmaDescSetSrcComp(dmaDesc, DESC_COMP_NONE);

  dmaDescSetROIWidth(dmaDesc, width);
  dmaDescSetROIHeight(dmaDesc, height);
  dmaDescSetSrcStride(dmaDesc, srcStride);
  dmaDescSetDstStride(dmaDesc, dstStride);

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
