//===- HexagonCAPI.h - hexagon alloc-free runtime calls -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// / The source pointer is a pointer to the base of memref.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_

#include "HexagonAPI.h"
extern "C" {
void *hexagon_runtime_alloc_1d(size_t bytes, uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_1d(void *ptr);
void *hexagon_runtime_alloc_2d(size_t numBlocks, size_t blockSize,
                               uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_2d(void *ptr);
void hexagon_runtime_copy(void *dst, void *src, size_t nbytes, bool isDVtcm,
                          bool isSrcVtcm);
}
#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
