//===- HexagonAPITests.cpp - THexagon API Tests                   ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonResources.h"

#include <gtest/gtest.h>

class HexagonAPITest : public ::testing::Test {
protected:
  void SetUp() { hexapi = HexagonAPI::Global(); }

public:
  static void SetUpTestSuite() { AllocateHexagonResources(); }
  static void TearDownTestSuite() { DeallocateHexagonResources(); }

  HexagonAPI *hexapi;
  size_t nbytes{2048};
  int64_t shape2d[2]{256, 256};
};

TEST_F(HexagonAPITest, global) { CHECK(hexapi != nullptr); }

TEST_F(HexagonAPITest, allocFree) {
  void *buf =
      hexapi->Alloc(nbytes, 128 /* alignment */, false /* isVtcm */); // int8
  CHECK(buf != nullptr);
  hexapi->Free(buf);
}

TEST_F(HexagonAPITest, allocndFreeVtcm) {
  void *buf1d = hexapi->Alloc(nbytes, 128 /* alignment */, true /* isVtcm */);
  CHECK(buf1d != nullptr);
  hexapi->Free(buf1d);

  void *buf2d = hexapi->Alloc(shape2d[0], nbytes, 2048 /* alignment */,
                              true) /* isVtcm */;
  CHECK(buf2d != nullptr);
  hexapi->Free(buf2d);
}

TEST_F(HexagonAPITest, allocScalar) {
  void *hexscalar = hexapi->Alloc(8, 128 /* alignment */, false /* isVtcm */);
  CHECK(hexscalar != nullptr);

  hexscalar = hexapi->Alloc(8, 128 /* alignment */, true /* isVtcm */);
  CHECK(hexscalar != nullptr);
}

TEST_F(HexagonAPITest, bufferManager) {
  void *bufferManager = hexapi->getBufferManager();
  CHECK(bufferManager != nullptr);
}

TEST_F(HexagonAPITest, vtcmPool) {
  VtcmPool *vtcmPool = hexapi->getVtcmPool();
  CHECK(vtcmPool != nullptr);
}

TEST_F(HexagonAPITest, copyVerification) {
  std::vector<uint8_t> data_in{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> data_out{7, 6, 5, 4, 3, 2, 1, 0};
  size_t n = data_in.size();
  void *buf1 = hexapi->Alloc(n, 128 /* alignment */, true /* isVtcm */); // int8
  void *buf2 = hexapi->Alloc(n, 128 /* alignment */, true /* isVtcm */); // int8
  uint8_t *ptr1 = static_cast<uint8_t *>(buf1);
  uint8_t *ptr2 = static_cast<uint8_t *>(buf2);
  hexapi->Copy(buf1, data_in.data(), n); // CopyFrom
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(ptr1[i], data_in[i]);
  }
  hexapi->Copy(data_out.data(), buf1, n); // CopyTo
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(data_out[i], data_in[i]);
  }
  hexapi->Copy(buf2, buf1, n); // CopyFrom HexagonBuffer
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(ptr2[i], ptr1[i]);
  }
}

// Buffer alias test removed - CreateBufferAlias and GetOrigBufferFromAlias
// functions no longer exist
