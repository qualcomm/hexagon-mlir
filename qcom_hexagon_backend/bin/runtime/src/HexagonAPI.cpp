//===- HexagonAPI.cpp -                                           ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// DataSpace: static allocations for Hexagon
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

#include "HexagonAPI.h"

HexagonAPI *HexagonAPI::Global() {
  static auto *inst = new HexagonAPI();
  return inst;
}
// DataSpace: static allocations for Hexagon
void *HexagonAPI::Alloc(size_t nbytes, uint64_t alignment, bool isVtcm) {
  // Allocate a single, contiguous memory region.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

uint32_t HexagonAPI::initialize_and_acquire_hmx() {
  HAP_power_request_t request;
  memset(&request, 0, sizeof(HAP_power_request_t));
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = TRUE;

  int *power_context = (int *)malloc(sizeof(int));

  int retVal = HAP_power_set((void *)power_context, &request);
  if (retVal != AEE_SUCCESS) {
    FARF(ERROR,
         "HMX resource allocation failed at HAP_power_set_HMX request with the "
         "return value %d",
         retVal);
    return retVal;
  }

  // Request HMX using HAP compute resource manager
  compute_res_attr_t compute_res;
  uint32_t contextID = 0;

  HAP_compute_res_attr_init(&compute_res);

  // Request the HMX resource
  HAP_compute_res_attr_set_hmx_param(&compute_res, 1);
  contextID = HAP_compute_res_acquire(&compute_res, 100000); // wait till 100ms

  if (contextID == 0) {
    return AEE_ERESOURCENOTFOUND;
  }

  // Locks the HMX unit to the current thread and prepares the thread to execute
  // HMX instructions
  HAP_compute_res_hmx_lock(contextID);

  return contextID;
}

void *HexagonAPI::Alloc(size_t nallocs, size_t nbytes, uint64_t alignment,
                        bool isVtcm) {
  // Allocate the region(s) needed for Hexagon's indirect-tensor format.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nallocs, nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

void HexagonAPI::Free(void *ptr) {
  if (bufferManager) {
    bufferManager->FreeHexagonBuffer(ptr);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}

// TODO: Add support to handle buffer aliases on copying
void HexagonAPI::Copy(void *dst, void *src, size_t nbytes) {
  if (bufferManager) {
    bufferManager->Copy(dst, src, nbytes);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}
