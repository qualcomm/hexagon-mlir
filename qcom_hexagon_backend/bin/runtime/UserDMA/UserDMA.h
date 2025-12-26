//===- UserDMA.h - Definition of UserDMA class ---------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// class UserDMA implements DMA APIs specified in RuntimeDMA.h that are wrapped
// in C style user facing APIs. Hexagon User DMA Engine's registers, descriptors
// and operations are abstracted in UserDMARegisters.h, UserDMADescriptors.h and
// UserDMAInstructions.h. This class uses a RingBuffer to hold a fixed number of
// DMA descriptors, and reuses entries in the ringbuffer for servicing DMA
// requests throughout its life, hence avoiding dynamic memory allocation. The
// 'copy' operation initiates a DMA transfer; it always tries to free an entry
// in the ring buffer before allocating a new entry for the transfer. The copy
// operation is usually non-blocking but blocks if the ring buffer is full. The
// copy operation returns a token that maps to an index in the ringbuffer. The
// wait operation blocks till the DMA transfer specified by the input token
// completes.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_USERDMA_H
#define HEXAGONBACKEND_BIN_RUNTIME_USERDMA_H
#include "RingBuffer.h"
#include "RuntimeDMA.h"
#include "UserDMADescriptors.h"
#include "UserDMAInstructions.h"
#include "UserDMARegisters.h"
#include <cassert>

namespace hexagon {
namespace userdma {
static const int MAX_NUM_DMA_DESCRIPTORS = 128;

class UserDMA {
public:
  UserDMA(int fifo_len);
  ~UserDMA();
  UserDMA(const UserDMA &) = delete;
  UserDMA &operator=(const UserDMA &) = delete;
  UserDMA(UserDMA &&) = delete;
  UserDMA &operator=(UserDMA &&) = delete;

  /*!
   * \brief Initiate DMA to copy memory from source to destination address
   * \param src Source address
   * \param srcAS Source address space
   * \param dst Destination address
   * \param dstAS Destination address space
   * \param length Length in bytes to copy
   * \param bypassCacheSrc should cache be bypassed when reading from src?
   * \param bypassCacheDst should cache be bypassed when writing to dst?
   * \returns a token that could be used for waiting on this DMA
   */
  uint32_t copy(void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS,
                uint32_t length, bool bypassCacheSrc, bool bypassCacheDst,
                DMAStatus *status);

  /*!
   * \brief Initiate 2D DMA to copy memory from source to destination address
   * \brief Some parameters have the same meaning as 1D DMA transfers (above)
   * \brief Description of other parameters below:
   * \param width Width of the 2D region
   * \param height Height of the 2D region
   * \param srcStride Source stride (distance in number of bytes between (x,y)
   and (x+1, y))
   * \param dstStride Destination stride
   * \param isOrdered If set to true, start of this DMA transfer will wait till
            all previous DMA transfers get completed
   * \param cacheAllocationPolicy:
      b00 - No read, No write allocation
      b01 - No read, write allocation
      b10 - Read allocation, no write allocation
      b11 - Read, Write allocation
   * \returns a token that could be used for waiting on this DMA transfer
  */
  uint32_t copy2D(void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS,
                  uint32_t width, uint32_t height, uint32_t srcStride,
                  uint32_t dstStride, bool bypassCacheSrc, bool bypassCacheDst,
                  bool isOrdered, uint32_t cacheAllocationPolicy,
                  DMAStatus *status);

  /*!
   * \brief Wait till the DMA transfer corresponding to the input token is
   * complete
   * \param token - token to wait
   */
  void wait(uint32_t token);

private:
  //! \brief Initializes the Hexagon User DMA engine
  unsigned int init();

  bool isFirstDMA = true; // Tracks if the next dma is the first DMA

  void *tailDMADesc = nullptr; // Tracks the tail DMA descriptor

  RingBuffer<DMADesc2D> *dmaQueue = nullptr; // Storage for all DMA descriptors
};

} // namespace userdma
} // namespace hexagon

#endif // HEXAGONBACKEND_BIN_RUNTIME_USERDMA_H
