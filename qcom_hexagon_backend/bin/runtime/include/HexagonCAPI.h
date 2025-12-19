#ifndef HEXAGONCAPI_H_
#define HEXAGONCAPI_H_

#include "HexagonAPI.h"
extern "C" {
void *hexagon_runtime_alloc_1d(size_t bytes, uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_1d(void *ptr);
void *hexagon_runtime_alloc_2d(size_t numBlocks, size_t blockSize,
                               uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_2d(void *ptr);
void hexagon_runtime_copy(void *dst, void *src, size_t nbytes, bool isDVtcm,
                          bool isSrcVtcm);

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton(void *source, size_t nbytes);
/// The source pointer is a pointer to crouton table
void *hexagon_rutime_get_contiguous_memref(void *source);
}
#endif // HEXAGONCAPI_H_
