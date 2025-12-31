//===-- wrapper.cpp -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <random>
#define FARF_ALWAYS 1
#include "CRunnerUtils.cpp"
#include "common.h"
#include "hexagon_benchmark.h"
#include "tensor.h"
#include "test_report.h"
#include <HAP_farf.h>
#include <cstdint>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

template <typename T> void rand_set(T *ptr, int size_in_bytes) {
  for (auto i = 0; i < size_in_bytes / sizeof(T); ++i)
    ptr[i] = static_cast<T>(rand() % 100);
  return;
}

// For each memref argument, we have these kernel arguments:
// char *allocated, char *aligned, int64_t offset, int64_t<num_dims> sizes,
// int64_t<num_dims> strides

using CopyFunc1 = void (*)(char *, char *, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, char *, char *, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t);
using CopyFunc2 = void (*)(char *, char *, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t, char *,
                           char *, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t);

extern "C" void hexmem_copy_firstTile_3d(char *, char *, int64_t, int64_t,
                                         int64_t, int64_t, int64_t, int64_t,
                                         int64_t, char *, char *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t,
                                         int64_t, int64_t);
extern "C" void dma_copy_firstTile_3d(char *, char *, int64_t, int64_t, int64_t,
                                      int64_t, int64_t, int64_t, int64_t,
                                      char *, char *, int64_t, int64_t, int64_t,
                                      int64_t, int64_t, int64_t, int64_t);
extern "C" void hexmem_copy_middleTile_3d(char *, char *, int64_t, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          int64_t, char *, char *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          int64_t, int64_t);
extern "C" void dma_copy_middleTile_3d(char *, char *, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t, char *, char *, int64_t,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t);
extern "C" void hexmem_copy_middleTile_4d(char *, char *, int64_t, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          int64_t, int64_t, int64_t, char *,
                                          char *, int64_t, int64_t, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          int64_t, int64_t);
extern "C" void dma_copy_middleTile_4d(char *, char *, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, char *,
                                       char *, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t);

Result run_test_3d(CopyFunc1 f, uint32_t channel, uint32_t height,
                   uint32_t width, const int64_t offset_c,
                   const int64_t offset_h, const int64_t offset_w,
                   const int64_t size_c, const int64_t size_h,
                   const int64_t size_w) {
  char *src = (char *)memalign(2048, channel * height * width * sizeof(char));
  char *dest = (char *)memalign(2048, channel * height * width * sizeof(char));

  rand_set<char>(src, channel * height * width * sizeof(char));
  memset(dest, 'X', channel * height * width * sizeof(char));

  f(src, src, 0, channel, height, width, height * width, width, 1, dest, dest,
    0, channel, height, width, height * width, width, 1);

  Result result = Result::Pass;

  // Simple verification: compare subview region
  for (int64_t c = offset_c; c < (offset_c + size_c); ++c) {
    for (int64_t i = offset_h; i < (offset_h + size_h); ++i) {
      for (int64_t j = offset_w; j < (offset_w + size_w); ++j) {
        int64_t src_idx = c * height * width + i * width + j;
        int64_t dest_idx = c * height * width + i * width + j;
        if (src[src_idx] != dest[dest_idx]) {
          printf("Mismatch at [%lld][%lld][%lld]: src=%c, dest=%c\n", c, i, j,
                 src[src_idx], dest[dest_idx]);
          result = Result::Fail;
          break;
        }
      }
    }
  }

  free(src);
  free(dest);
  return result;
}

Result run_test_4d(CopyFunc2 f, uint32_t channel1, uint32_t channel2,
                   uint32_t height, uint32_t width, const int64_t offset_c1,
                   const int64_t offset_c2, const int64_t offset_h,
                   const int64_t offset_w, const int64_t size_c1,
                   const int64_t size_c2, const int64_t size_h,
                   const int64_t size_w) {
  char *src = (char *)memalign(2048, channel1 * channel2 * height * width *
                                         sizeof(char));
  char *dest = (char *)memalign(2048, channel1 * channel2 * height * width *
                                          sizeof(char));

  rand_set<char>(src, channel1 * channel2 * height * width * sizeof(char));
  memset(dest, 'X', channel1 * channel2 * height * width * sizeof(char));

  f(src, src, 0, channel1, channel2, height, width, channel2 * height * width,
    height * width, width, 1, dest, dest, 0, channel1, channel2, height, width,
    channel2 * height * width, height * width, width, 1);

  Result result = Result::Pass;

  // Simple verification: compare subview region
  for (int64_t c1 = offset_c1; c1 < (offset_c1 + size_c1); ++c1) {
    for (int64_t c2 = offset_c2; c2 < (offset_c2 + size_c2); ++c2) {
      for (int64_t i = offset_h; i < (offset_h + size_h); ++i) {
        for (int64_t j = offset_w; j < (offset_w + size_w); ++j) {
          int64_t src_idx = c1 * channel2 * height * width +
                            c2 * height * width + i * width + j;
          int64_t dest_idx = c1 * channel2 * height * width +
                             c2 * height * width + i * width + j;
          if (src[src_idx] != dest[dest_idx]) {
            printf("Mismatch at [%lld][%lld][%lld][%lld]: src=%c, dest=%c\n",
                   c1, c2, i, j, src[src_idx], dest[dest_idx]);
            result = Result::Fail;
            break;
          }
        }
      }
    }
  }

  free(src);
  free(dest);
  return result;
}

int main(int argc, char **argv) {
  Result result1 =
      run_test_3d(hexmem_copy_firstTile_3d, 2, 64, 64, 0, 0, 0, 2, 32, 32);
  if (result1 != Result::Pass) {
    FARF(ALWAYS, "hexmem_copy_firstTile_3d failed\n");
    return result1;
  }

  Result result2 =
      run_test_3d(dma_copy_firstTile_3d, 2, 64, 64, 0, 0, 0, 2, 32, 32);
  if (result2 != Result::Pass) {
    FARF(ALWAYS, "dma_copy_firstTile_3d Failed\n");
    return result2;
  }

  if (result1 != result2) {
    FARF(ALWAYS,
         "hexmem_copy_firstTile_3d and dma_copy_firstTile_3d does not match\n");
  }

  result1 = run_test_3d(hexmem_copy_middleTile_3d, 10, 128, 128, 4, 64, 64, 2,
                        32, 32);
  if (result1 != Result::Pass) {
    FARF(ALWAYS, "hexmem_copy_middleTile_3d failed\n");
    return result1;
  }

  result2 =
      run_test_3d(dma_copy_middleTile_3d, 10, 128, 128, 4, 64, 64, 2, 32, 32);
  if (result2 != Result::Pass) {
    FARF(ALWAYS, "dma_copy_middleTile_3d Failed\n");
    return result2;
  }

  if (result1 != result2) {
    FARF(ALWAYS, "hexmem_copy_middleTile_3d and dma_copy_middleTile_3d does "
                 "not match\n");
  }

  result1 = run_test_4d(hexmem_copy_middleTile_4d, 20, 10, 128, 128, 8, 4, 64,
                        64, 4, 2, 32, 32);
  if (result1 != Result::Pass) {
    FARF(ALWAYS, "hexmem_copy_middleTile_4d failed\n");
    return result1;
  }

  result2 = run_test_4d(dma_copy_middleTile_4d, 20, 10, 128, 128, 8, 4, 64, 64,
                        4, 2, 32, 32);
  if (result2 != Result::Pass) {
    FARF(ALWAYS, "dma_copy_middleTile_4d Failed\n");
    return result2;
  }

  if (result1 != result2) {
    FARF(ALWAYS, "hexmem_copy_middleTile_4d and dma_copy_middleTile_4d does "
                 "not match\n");
  }

  return result2;
}
