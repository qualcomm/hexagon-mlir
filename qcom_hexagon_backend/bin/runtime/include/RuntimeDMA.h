//=== RuntimeDMA.h - specification of DMA APIs for users ===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_RUNTIME_DMA_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_RUNTIME_DMA_H

#include <cstdint>

namespace hexagon {
namespace userdma {

typedef enum { DMAFailure = -1, DMASuccess = 0 } DMAStatus;
typedef enum { DDR = 0, VTCM = 1 } AddrSpace;

extern "C" uint32_t
hexagon_runtime_dma_start(void *src, AddrSpace srcAS, void *dst,
                          AddrSpace dstAS, uint32_t length, bool bypassCacheSrc,
                          bool bypassCacheDst, DMAStatus *status);

extern "C" void hexagon_runtime_dma_wait(uint32_t token);

} // namespace userdma
} // namespace hexagon

#endif
