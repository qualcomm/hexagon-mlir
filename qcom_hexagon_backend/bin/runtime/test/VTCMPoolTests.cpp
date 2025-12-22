//===- VTCMPoolTests.cpp - implementation file                       ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonCommon.h"
#include "HexagonResources.h"
#include "VTCMPool.h"

#include <gtest/gtest.h>

class VtcmPoolTest : public ::testing::Test {
  void SetUp() override {
    vtcm_pool = HexagonAPI::Global()->getVtcmPool();
    max_bytes = vtcm_pool->VtcmAllocatedBytes();
    device_bytes = vtcm_pool->VtcmDeviceBytes();
  }

public:
  static void SetUpTestSuite() { AllocateHexagonResources(); }
  static void TearDownTestSuite() { DeallocateHexagonResources(); }

  VtcmPool *vtcm_pool;
  size_t max_bytes;
  size_t device_bytes;
  size_t four_k_block = 4096;
  size_t two_k_block = 2048;
  size_t one_k_block = 1024;
  size_t min_bytes = 128;
};

TEST_F(VtcmPoolTest, basic) {
  void *ptr;
  void *ptr2;

  ptr = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr, max_bytes);

  ptr = vtcm_pool->Allocate(two_k_block);
  vtcm_pool->Free(ptr, two_k_block);

  ptr = vtcm_pool->Allocate(one_k_block);
  vtcm_pool->Free(ptr, one_k_block);

  ptr = vtcm_pool->Allocate(min_bytes);

  ptr2 = vtcm_pool->Allocate(one_k_block);
  vtcm_pool->Free(ptr, min_bytes);
  vtcm_pool->Free(ptr2, one_k_block);
}

TEST_F(VtcmPoolTest, small_allocations) {
  void *ptr1;
  void *ptr2;
  void *ptr3;
  void *ptr4;

  // Allocate small chunk from the back
  ptr1 = vtcm_pool->Allocate(min_bytes);

  // Allocate from the front
  ptr2 = vtcm_pool->Allocate(two_k_block);

  // Allocate the rest
  ptr3 = vtcm_pool->Allocate(max_bytes - min_bytes - two_k_block);

  vtcm_pool->Free(ptr1, min_bytes);
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, max_bytes - min_bytes - two_k_block);

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}

TEST_F(VtcmPoolTest, free_alloc_combinations) {
  void *ptr1;
  void *ptr2;
  void *ptr3;
  void *ptr4;
  void *new_ptr;
  size_t max_less_3_blocks = max_bytes - (3 * two_k_block);
  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(two_k_block);
  ptr4 = vtcm_pool->Allocate(max_less_3_blocks);

  // Make sure pointers are 2k apart from each other
  CHECK(static_cast<char *>(ptr1) + two_k_block == static_cast<char *>(ptr2));
  CHECK(static_cast<char *>(ptr2) + two_k_block == static_cast<char *>(ptr3));
  CHECK(static_cast<char *>(ptr3) + two_k_block == static_cast<char *>(ptr4));

  // Free 2, realloc it, make sure it is the same as before
  vtcm_pool->Free(ptr2, two_k_block);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  // Free 1 and 2, re-alloc and make sure they are the same
  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr1);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  // Exercise different deletion scenarios
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr4, max_less_3_blocks);
  vtcm_pool->Free(ptr1, two_k_block);

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(two_k_block);
  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}

TEST_F(VtcmPoolTest, find_allocation) {
  void *ptr1;
  void *ptr2;
  void *ptr3;

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);

  // Free the first allocation
  vtcm_pool->Free(ptr1, two_k_block);

  // Allocate a new larger block to initiate search and ensure
  // it succeeds despite there not being a match in the first free block.
  ptr3 = vtcm_pool->Allocate(four_k_block);

  // Clean up the ptrs
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, four_k_block);

  // Make sure at the end we have the full amount available again
  ptr1 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr1, max_bytes);
}

TEST_F(VtcmPoolTest, find_smallest_allocation_combinations) {
  void *ptr1;
  void *ptr2;
  void *ptr3;
  void *ptr4;
  void *new_ptr;

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(four_k_block);
  ptr4 = vtcm_pool->Allocate(four_k_block);

  // Fragment memory allocations.
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, four_k_block);

  // Reallocate memory allocations and ensure that the smallest free allocations
  // are used.
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr3);

  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr4, four_k_block);

  // Rerun the same test for non 2k aligned allocations.
  ptr1 = vtcm_pool->Allocate(min_bytes);
  ptr2 = vtcm_pool->Allocate(min_bytes);
  ptr3 = vtcm_pool->Allocate(one_k_block);
  ptr4 = vtcm_pool->Allocate(one_k_block);

  // Fragment memory allocations.
  vtcm_pool->Free(ptr2, min_bytes);
  vtcm_pool->Free(ptr3, one_k_block);

  // Reallocate memory allocations and ensure that the smallest free allocations
  // are used.
  new_ptr = vtcm_pool->Allocate(min_bytes);
  CHECK(new_ptr == ptr2);

  new_ptr = vtcm_pool->Allocate(one_k_block);
  CHECK(new_ptr == ptr3);

  vtcm_pool->Free(ptr1, min_bytes);
  vtcm_pool->Free(ptr2, min_bytes);
  vtcm_pool->Free(ptr3, one_k_block);
  vtcm_pool->Free(ptr4, one_k_block);

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}
