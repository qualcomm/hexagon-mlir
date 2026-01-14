//===- HexagonBufferTests.cpp - implmentation file.              -----------==//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonBuffer.h"
#include "HexagonResources.h"
#include <gtest/gtest.h>

class HexagonBufferTest : public ::testing::Test {
public:
  static void SetUpTestSuite() { AllocateHexagonResources(); }
  static void TearDownTestSuite() { DeallocateHexagonResources(); }
};

TEST_F(HexagonBufferTest, defaultScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST_F(HexagonBufferTest, ddrScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST_F(HexagonBufferTest, vtcmScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, true);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, vtcmScope2) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, true);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, microCopiesCorrespondingRegions) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 2);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(16 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 16 * i));
    EXPECT_EQ(microCopies[i].numBytes, 16);
  }
}

TEST_F(HexagonBufferTest, microCopiesSrcBigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(72), ptr(80), ptr(88)};
  BufferSet dest(destPtr.data(), destPtr.size(), 8);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 4);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(8 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(microCopies[i].numBytes, 8);
  }
}

TEST_F(HexagonBufferTest, microCopiesDestBigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(8), ptr(16), ptr(24)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 8);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 4);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(8 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(microCopies[i].numBytes, 8);
  }
}

TEST_F(HexagonBufferTest, microCopies_src_overlaps_dest_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(76)};
  BufferSet dest(destPtr.data(), destPtr.size(), 12);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 12);

  // First region of source, second region of dest
  EXPECT_EQ(microCopies[1].src, ptr(12));
  EXPECT_EQ(microCopies[1].dest, ptr(76));
  EXPECT_EQ(microCopies[1].numBytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(16));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 8);
}

TEST_F(HexagonBufferTest, microCopies_dest_overlaps_src_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(12)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 12);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 12);

  // Second region of source, first region of dest
  EXPECT_EQ(microCopies[1].src, ptr(12));
  EXPECT_EQ(microCopies[1].dest, ptr(76));
  EXPECT_EQ(microCopies[1].numBytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(16));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 8);
}

TEST_F(HexagonBufferTest, microCopiesDiscontiguousRegions) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  // Stride of 16, but only first 11 bytes in each region belong to
  // this buffer.
  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 11);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 13);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 16);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 11);

  // Second region of source, first region of dest
  EXPECT_EQ(microCopies[1].src, ptr(16));
  EXPECT_EQ(microCopies[1].dest, ptr(75));
  EXPECT_EQ(microCopies[1].numBytes, 2);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(18));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 3);
}

TEST_F(HexagonBufferTest, copyFrom) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t *ptr = static_cast<uint8_t *>(hb.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, nd) {
  HexagonBuffer hb_default(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                           false /* isVTCM */);
  EXPECT_EQ(hb_default.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  HexagonBuffer hbGlobal(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                         false /* isVTCM */);
  EXPECT_EQ(hbGlobal.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  HexagonBuffer hbVtcm(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                       true /* isVTCM */);
  EXPECT_EQ(hbVtcm.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, nd_copy_from) {
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                   false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t **ptr = static_cast<uint8_t **>(hb.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST_F(HexagonBufferTest, 1dCopyFrom1d) {
  HexagonBuffer from(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);
  HexagonBuffer to(8 /* nbytes */, 8 /* alignment */, true /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  from.CopyFrom(data.data(), data.size());
  to.CopyFrom(from, 8);

  uint8_t *ptr = static_cast<uint8_t *>(to.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, 2d_copy_from_1d) {
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, true /* isVTCM */);
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb1d.CopyFrom(data.data(), data.size());
  hb2d.CopyFrom(hb1d, 8);

  uint8_t **ptr = static_cast<uint8_t **>(hb2d.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST_F(HexagonBufferTest, 1dCopyFrom2d) {
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     true /* isVTCM */);
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb2d.CopyFrom(data.data(), data.size());
  hb1d.CopyFrom(hb2d, 8);

  uint8_t *ptr = static_cast<uint8_t *>(hb1d.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, mdCopyFromNd) {
  HexagonBuffer hb3d(3 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);
  HexagonBuffer hb4d(4 /* ndim */, 3 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  hb3d.CopyFrom(data.data(), data.size());
  hb4d.CopyFrom(hb3d, data.size());

  uint8_t **hb3dPtr = static_cast<uint8_t **>(hb3d.GetPointer());
  uint8_t **hb4dPtr = static_cast<uint8_t **>(hb4d.GetPointer());
  for (size_t i = 0; i < 12; i++) {
    EXPECT_EQ(hb3dPtr[i / 4][i % 4], hb4dPtr[i / 3][i % 3]);
  }
}

TEST_F(HexagonBufferTest, copyTo) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> dataIn{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(dataIn.data(), dataIn.size());

  std::vector<uint8_t> dataOut{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(dataOut.data(), dataOut.size());

  for (size_t i = 0; i < dataIn.size(); ++i) {
    EXPECT_EQ(dataIn[i], dataOut[i]);
  }
}

TEST_F(HexagonBufferTest, ndCopyTo) {
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                   false /* isVTCM */);

  std::vector<uint8_t> dataIn{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(dataIn.data(), dataIn.size());

  std::vector<uint8_t> dataOut{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(dataOut.data(), dataOut.size());

  for (size_t i = 0; i < dataIn.size(); ++i) {
    EXPECT_EQ(dataIn[i], dataOut[i]);
  }
}
