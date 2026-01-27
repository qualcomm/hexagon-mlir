//===- HexagonCAPITests.cpp                                  --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonCAPI.h"
#include "HexagonResources.h"
#include <gtest/gtest.h>

TEST(HexagonCAPItest, test_flow) {
  AllocateHexagonResources();
  size_t bytes{2048};
  int64_t shape2d[2]{256, 256};
  size_t alignment{128};
  size_t alignment_2d{2048};
  void *buf1 = hexagon_runtime_alloc_1d(bytes, alignment, false /* isVtcm */);
  void *buf2 = hexagon_runtime_alloc_1d(bytes, alignment, true /* isVtcm */);

  hexagon_runtime_free_1d(buf1);
  hexagon_runtime_free_1d(buf2);
  buf1 = hexagon_runtime_alloc_2d(shape2d[0], bytes, alignment_2d,
                                  false /* isVtcm */);
  buf2 = hexagon_runtime_alloc_2d(shape2d[0], bytes, alignment_2d,
                                  true /* isVtcm */);
  hexagon_runtime_free_2d(buf1);
  hexagon_runtime_free_2d(buf2);
  DeallocateHexagonResources();
}
