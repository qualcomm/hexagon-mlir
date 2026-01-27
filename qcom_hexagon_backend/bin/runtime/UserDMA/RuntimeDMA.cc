//===- RuntimeDMA.cc - DMA APIs for users ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The Hexagon User DMA hardware allows users to have low level controls over
// cache behaviour during DMA transfers. It is possible to avoid cache actions
// for enforcing coherency by programming the "bypassCache" bits. It is
// possible for the Hexagon hardware to check if a memory location is modified
// by other hardware units (eg. DMA) and if so, to mark cache line for that
// location, if present in the cache, as invalid (stale). This "snooping" could
// cost power and time. If the programmer is sure that the write location is
// absent in cache (eg. that location is mapped to VTCM), this 'snoop and
// invalidate' action is redundant. We can prevent this redundant effort using
// the "bypassCache" flag (we can specify bypass cache options - details in the
// API description). This optimization should be used carefully, or data
// coherence could be hit.
//
//===----------------------------------------------------------------------===//

#include "RuntimeDMA.h"
#include "UserDMA.h"

namespace hexagon {
namespace userdma {

// Create a static object of HexagonUserDMA to be used in DMAStart and
// DMAWait APIs. TODO: Make HexagonUserDMA class as a singleton and
// call its static member function to get its unique instance
static UserDMA hexDMA(hexagon::userdma::MAX_NUM_DMA_DESCRIPTORS);

/*!
 * Initiate DMA to copy memory from source to destination address
 * src: Source address
 * srcAS: Source address space
 * dst: Destination address
 * dstAS: Destination address space
 * length: Length in bytes to copy
 * bypassCacheSrc: if set to true, HW cache will be bypassed when reading from
 * memory bypassCacheDst: if set to true, HW cache will be bypassed when writing
 * to memory returns a token that could be used for waiting on this DMA
 */
extern "C" uint32_t
hexagon_runtime_dma_start(void *src, AddrSpace srcAS, void *dst,
                          AddrSpace dstAS, uint32_t length, bool bypassCacheSrc,
                          bool bypassCacheDst, DMAStatus *status) {
  return hexDMA.copy(src, srcAS, dst, dstAS, length, bypassCacheSrc,
                     bypassCacheDst, status);
}

/*!
 * Initiate DMA to copy 2D memory from source to destination address
 * The assumption here is that src and dst are contiguous (stride 1 in innermost
 * dim).
 * src: Source address
 * srcAS: Source address space
 * dst: Destination address
 * dstAS: Destination address space
 * width: width of 2D region to copy
 * height: height of 2D region to copy (num bytes = width * height)
 * srcStride: stride for source
 * dstStride: stride for destination
 * bypassCacheSrc: if set to true, HW cache will be bypassed when reading from
 memory
 * bypassCacheDst: if set to true, HW cache will be bypassed when writing to
 memory
 * isOrdered: if set to true, this DMA will be initiated after all previous
 *   DMA transfers initiated earlier complete
 * cacheAllocationPolicy:
    b00: No Read/Write allocation
    b01: Write allocation
    b10: Read allocation
    b11: Read/Write allocation
 * status: address of location to which status of DMA initiation will be written
 * returns a token that could be used for waiting on this DMA
 */
extern "C" uint32_t hexagon_runtime_dma2d_start(
    void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS, uint32_t width,
    uint32_t height, uint32_t srcStride, uint32_t dstStride,
    bool bypassCacheSrc, bool bypassCacheDst, bool isOrdered,
    uint32_t cacheAllocationPolicy, DMAStatus *status) {
  return hexDMA.copy2D(src, srcAS, dst, dstAS, width, height, srcStride,
                       dstStride, bypassCacheSrc, bypassCacheDst, isOrdered,
                       cacheAllocationPolicy, status);
}

/*!
 * Wait till the DMA transfer corresponding to the input token is
 * complete
 * token: token corresponding to the DMA transfer to wait
 */
extern "C" void hexagon_runtime_dma_wait(uint32_t token) { hexDMA.wait(token); }

} // namespace userdma
} // namespace hexagon
