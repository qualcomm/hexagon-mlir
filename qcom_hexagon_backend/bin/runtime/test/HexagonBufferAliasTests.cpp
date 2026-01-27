//===- HexagonBufferAliasTests.cpp                     --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonBuffer.h"
#include "HexagonBufferAlias.h"
#include "HexagonResources.h"
#include <gtest/gtest.h>

class HexagonBufferAliasTest : public ::testing::Test {
public:
  static void SetUpTestSuite() { AllocateHexagonResources(); }
  static void TearDownTestSuite() { DeallocateHexagonResources(); }
};

TEST_F(HexagonBufferAliasTest, CreateAlias) {
  HexagonBuffer hb(2 * 2048 /* nbytes */, 2048 /* alignment */, false);
  auto alias = hexagon::HexagonBufferAlias(hb, 2 * 2048);
  void **pointerTableBasePtr =
      reinterpret_cast<void **>(alias.GetPointerTableBase());
  uint8_t *pointerPtr = reinterpret_cast<uint8_t *>(*pointerTableBasePtr);
  uint8_t *bufferPtr = reinterpret_cast<uint8_t *>(hb.GetPointer());
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(pointerPtr, bufferPtr);
    pointerPtr += 2048;
    bufferPtr += 2048;
  }
}

TEST_F(HexagonBufferAliasTest, CreateAliasVtcm) {
  HexagonBuffer hb(2 * 2048 /* nbytes */, 2048 /* alignment */, true);
  auto alias = hexagon::HexagonBufferAlias(hb, 2 * 2048);
  void **pointerTableBasePtr =
      reinterpret_cast<void **>(alias.GetPointerTableBase());
  uint8_t *pointerPtr = reinterpret_cast<uint8_t *>(*pointerTableBasePtr);
  uint8_t *bufferPtr = reinterpret_cast<uint8_t *>(hb.GetPointer());
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(pointerPtr, bufferPtr);
    pointerPtr += 2048;
    bufferPtr += 2048;
  }
}
