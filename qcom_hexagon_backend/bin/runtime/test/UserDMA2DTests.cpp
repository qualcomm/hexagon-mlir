//===- UserDMA2DTests.cpp - Unit tests for 2D DMAStart and Wait   ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The test creates and fills a "2D" int8 array with random values;
// it then creates 4 smaller 2D tensors (each corresponding to a tile
// of the bigger tensor), and initiates 2D DMA copies from the bigger
// tensor to each of the smaller tensors. Finally it checks that the
// smaller tensors have the expected values.
// Limitations of the implementation:
// 1. DMA runtime API implementation is not thread-safe
// 2. API implementation frees up entries in the DMA descriptor FIFO
// only after the FIFO becomes full. It should be modified so that
// FIFO entries could be freed earlier as well (when possible)
// Limitations of the test:
// 1. The test checks only the DMA runtime API
// 2. It doesn't check for erroneous returns from API
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
template <typename T> void randSet2D(T *ptr, int sizeInBytes) {
  for (auto i = 0; i < sizeInBytes / sizeof(T); ++i)
    ptr[i] = static_cast<T>(rand());
  return;
}

// Demonstrate a test which exercises 2D DMA transfers involving char
// A 400X400 tensor is tiled to four 200X200 tiles
// Copy tile data in original 400X400 tensor to smaller tensor tiles using DMA
TEST(UDMATest2D, FourTensorTileCopy) {
  const int ORIG_HEIGHT = 400;
  const int ORIG_WIDTH = 400;
  const int TILE_HEIGHT = 200;
  const int TILE_WIDTH = 200;
  const int ORIG_STRIDE = ORIG_WIDTH;
  const int ORIG_SIZE = ORIG_HEIGHT * ORIG_WIDTH;
  const int TILE_SIZE = TILE_HEIGHT * TILE_WIDTH;
  const int NUM_TILES_ALONG_WIDTH = ORIG_WIDTH / TILE_WIDTH;

  char *src0 = (char *)malloc(ORIG_SIZE);
  char *tile0 = (char *)malloc(TILE_SIZE);
  char *tile1 = (char *)malloc(TILE_SIZE);
  char *tile2 = (char *)malloc(TILE_SIZE);
  char *tile3 = (char *)malloc(TILE_SIZE);

  char *refTile0 = (char *)malloc(TILE_SIZE);
  char *refTile1 = (char *)malloc(TILE_SIZE);
  char *refTile2 = (char *)malloc(TILE_SIZE);
  char *refTile3 = (char *)malloc(TILE_SIZE);

  char *refTiles[4] = {refTile0, refTile1, refTile2, refTile3};

  randSet2D<char>(src0, ORIG_SIZE);
  randSet2D<char>(tile0, TILE_SIZE);
  randSet2D<char>(tile1, TILE_SIZE);
  randSet2D<char>(tile2, TILE_SIZE);
  randSet2D<char>(tile3, TILE_SIZE);

  for (int i = 0; i < ORIG_HEIGHT; ++i) {
    for (int j = 0; j < ORIG_WIDTH; ++j) {
      // Orig element (i, j) goes to tile (i / TILE_HEIGHT, j / TILE_WIDTH)
      // In that tile, it goes to tile offset (i % TILE_HEIGHT, j % TILE_WIDTH)
      // Use the above 2-d tile id, 2-dtile offset to compute flat tile_id, flat
      // tile_offset
      int flatTileID =
          (i / TILE_HEIGHT) * NUM_TILES_ALONG_WIDTH + (j / TILE_WIDTH);
      int offsetInTile = (i % TILE_HEIGHT) * TILE_WIDTH + (j % TILE_WIDTH);
      refTiles[flatTileID][offsetInTile] = src0[i * ORIG_STRIDE + j];
    }
  }

  hexagon::userdma::DMAStatus status;
  hexagon::userdma::UserDMA hexDMASmallQCapacity(
      2); // A small queue to test the case when DMA queue becomes full
  uint32_t token0 = hexDMASmallQCapacity.copy2D(
      src0, hexagon::userdma::DDR, tile0, hexagon::userdma::DDR, TILE_WIDTH,
      TILE_HEIGHT, ORIG_STRIDE, TILE_WIDTH, false, false, false, 0, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  uint32_t token1 = hexDMASmallQCapacity.copy2D(
      &(src0[TILE_WIDTH]), hexagon::userdma::DDR, tile1, hexagon::userdma::DDR,
      TILE_WIDTH, TILE_HEIGHT, ORIG_STRIDE, TILE_WIDTH, false, false, false, 0,
      &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  uint32_t token2 = hexDMASmallQCapacity.copy2D(
      &(src0[TILE_HEIGHT * ORIG_STRIDE]), hexagon::userdma::DDR, tile2,
      hexagon::userdma::DDR, TILE_WIDTH, TILE_HEIGHT, ORIG_STRIDE, TILE_WIDTH,
      false, false, false, 0, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);
  uint32_t token3 = hexDMASmallQCapacity.copy2D(
      &(src0[TILE_HEIGHT * ORIG_STRIDE + TILE_WIDTH]), hexagon::userdma::DDR,
      tile3, hexagon::userdma::DDR, TILE_WIDTH, TILE_HEIGHT, ORIG_STRIDE,
      TILE_WIDTH, false, false, false, 0, &status);
  EXPECT_EQ(status, hexagon::userdma::DMASuccess);

  hexDMASmallQCapacity.wait(token0);
  hexDMASmallQCapacity.wait(token1);
  hexDMASmallQCapacity.wait(token2);
  hexDMASmallQCapacity.wait(token3);

  EXPECT_EQ(std::memcmp(refTile0, tile0, TILE_SIZE), 0);
  EXPECT_EQ(std::memcmp(refTile1, tile1, TILE_SIZE), 0);
  EXPECT_EQ(std::memcmp(refTile2, tile2, TILE_SIZE), 0);
  EXPECT_EQ(std::memcmp(refTile3, tile3, TILE_SIZE), 0);
}
