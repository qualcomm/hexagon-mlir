//===- HexagonCAPI.cpp -  hexagon alloc free runtime calls ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// / The source pointer is a pointer to the base of memref
//
//===----------------------------------------------------------------------===//
#include "HexagonCAPI.h"
#include "HexagonCommon.h"
#include <cassert>

extern "C" {
void *hexagon_runtime_alloc_1d(size_t bytes, uint64_t alignment, bool isVtcm) {
  return HexagonAPI::Global()->Alloc(bytes, alignment, isVtcm);
}

void *hexagon_runtime_alloc_2d(size_t numBlocks, size_t blockSize,
                               uint64_t alignment, bool isVtcm) {
  return HexagonAPI::Global()->Alloc(numBlocks, blockSize, alignment, isVtcm);
}

void hexagon_runtime_free_1d(void *ptr) { HexagonAPI::Global()->Free(ptr); }

void hexagon_runtime_free_2d(void *ptr) { HexagonAPI::Global()->Free(ptr); }

void hexagon_runtime_copy(void *dst, void *src, size_t nbytes, bool isDstVtcm,
                          bool isSrcVtcm) {
  HexagonAPI::Global()->Copy(dst, src, nbytes);
}

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton(void *source, size_t nbytes) {
  assert(nbytes % CROUTON_SIZE == 0 &&
         "The size is expected to be a multiple of crouton size");
  return HexagonAPI::Global()->CreateBufferAlias(source, nbytes);
}

/// The source pointer is a pointer to crouton table
void *hexagon_runtime_get_contiguous_memref(void *source) {
  return HexagonAPI::Global()->GetOrigBufferFromAlias(source);
}
}
