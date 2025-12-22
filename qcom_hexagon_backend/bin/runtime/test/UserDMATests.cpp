//===- UserDMATests.cpp - Unit tests for DMAStart and Wait     ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Fill the input array (ptr) with "random" bytes. The test checks that those
// "random" values are copied correctly to the destination tensor.
//
//===----------------------------------------------------------------------===//

#include "UserDMA.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <unistd.h>

// Fill the input array (ptr) with "random" bytes. The test checks that those
// "random" values are copied correctly to the destination tensor.
template <typename T> void rand_set(T *ptr, int size_in_bytes) {
  for (auto i = 0; i < size_in_bytes / sizeof(T); ++i)
    ptr[i] = static_cast<T>(rand());
  return;
}

template <> void rand_set<float>(float *ptr, int size_in_bytes) {
  // Generate uniform random floats between -RAND_MAX and RAND_MAX
  std::random_device rand_dev;
  std::default_random_engine rand_eng(rand_dev());
  std::uniform_real_distribution<float> dis(static_cast<float>(-RAND_MAX),
                                            static_cast<float>(RAND_MAX));
  for (auto i = 0; i < size_in_bytes / sizeof(float); ++i)
    ptr[i] = dis(rand_eng);
  return;
}

// Demonstrate a test which exercises DMA transfers involving char, int
// and float data types.
TEST(UDMATest, ThreeTensorCopy) {
  const int SRC0_SIZE =
      16777215; // 16M-1 is the upper bound for transfer size (bytes)
  const int SRC1_SIZE = 4194303; // Exercise max sized transfer
  const int SRC2_SIZE = 4194303;

  char *src0 = (char *)malloc(SRC0_SIZE);
  int *src1 = (int *)malloc(SRC1_SIZE * sizeof(int));
  float *src2 = (float *)malloc(SRC2_SIZE * sizeof(float));

  char *dst0 = (char *)malloc(SRC0_SIZE);
  int *dst1 = (int *)malloc(SRC1_SIZE * sizeof(int));
  float *dst2 = (float *)malloc(SRC2_SIZE * sizeof(float));

  rand_set<char>(src0, SRC0_SIZE);
  rand_set<int>(src1, SRC1_SIZE * sizeof(int));
  rand_set<float>(src2, SRC2_SIZE * sizeof(float));

  hexagon::userdma::DMAStatus status;
  hexagon::userdma::UserDMA hexDMASmallQCapacity(
      2); // A small queue to test the case when DMA queue becomes full
  // Try to fill the queue with 3 huge asynchronous transfers
  uint32_t token0 = hexDMASmallQCapacity.copy(src0, hexagon::userdma::DDR, dst0,
                                              hexagon::userdma::DDR, SRC0_SIZE,
                                              false, false, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  uint32_t token1 = hexDMASmallQCapacity.copy(
      src1, hexagon::userdma::DDR, dst1, hexagon::userdma::DDR,
      SRC1_SIZE * sizeof(int), false, false, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  uint32_t token2 = hexDMASmallQCapacity.copy(
      src2, hexagon::userdma::DDR, dst2, hexagon::userdma::DDR,
      SRC2_SIZE * sizeof(float), false, false, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);

  hexDMASmallQCapacity.wait(token0);
  hexDMASmallQCapacity.wait(token1);
  hexDMASmallQCapacity.wait(token2);

  EXPECT_EQ(std::memcmp(src0, dst0, SRC0_SIZE), 0);
  EXPECT_EQ(std::memcmp(src1, dst1, SRC1_SIZE * sizeof(int)), 0);
  EXPECT_EQ(std::memcmp(src2, dst2, SRC2_SIZE * sizeof(float)), 0);

  // Apparently "Synchronous" transfer - copy and wait back-to-back
  // TODO: Move this synchronous transfer case as a separate test once our
  // runtime becomes thread-safe
  rand_set<char>(src0, SRC0_SIZE);
  token0 = hexDMASmallQCapacity.copy(src0, hexagon::userdma::DDR, dst0,
                                     hexagon::userdma::DDR, SRC0_SIZE, false,
                                     false, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  hexDMASmallQCapacity.wait(token0);
  EXPECT_EQ(std::memcmp(src0, dst0, SRC0_SIZE), 0);
}
