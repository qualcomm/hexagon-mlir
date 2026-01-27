//===- HexagonAPI.h - device API   to compile run on Hexagon --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Hexagon Device API that is compiled and run on Hexagon.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_
#define HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HAP_compute_res.h"
#include "HAP_power.h"

#include "BufferManager.h"
#include "HexagonCommon.h"
#include "VTCMPool.h"

/// Hexagon Device API that is compiled and run on Hexagon.
class HexagonAPI {
public:
  /// Retrieve the global singleton instance of the HexagonAPI.
  static HexagonAPI *Global();

  /// Constructor
  HexagonAPI() { this->AcquireResources(); }

  /// Destructor
  ~HexagonAPI() { this->ReleaseResources(); }

  /// Ensures resource managers are in a good state for the runtime
  void AcquireResources() {
    CHECK_EQ(runtimeVtcm, nullptr);
    runtimeVtcm = std::make_unique<VtcmPool>();

    CHECK_EQ(bufferManager, nullptr);
    bufferManager = std::make_unique<BufferManager>();

    hmx_context_id = initialize_and_acquire_hmx();
  }

  /// Ensures all runtime resources are freed
  void ReleaseResources() {
    CHECK((bufferManager), "bufferManager was not created in AcquireResources");
    bufferManager.reset();

    CHECK((runtimeVtcm), "runtimeVtcm was not created in AcquireResources");
    runtimeVtcm.reset();

    release_hmx();
  }

  /// Returns VtcmPool instance
  VtcmPool *getVtcmPool() {
    CHECK((runtimeVtcm), "runtimeVtcm has not been created");
    return runtimeVtcm.get();
  }

  /// Returns BufferManager instance
  BufferManager *getBufferManager() {
    CHECK((bufferManager), "bufferManager has not been created");
    return bufferManager.get();
  }

  /// Returns True, if the resources are initialized
  bool hasResources() {
    if (runtimeVtcm == nullptr || bufferManager == nullptr)
      return false;
    return true;
  }

  /// Allocate a single, contiguous memory region.
  void *Alloc(size_t nbytes, uint64_t alignment, bool isVtcm);

  /// Allocate the region(s) needed for Hexagon's indirect-tensor format.
  void *Alloc(size_t nallocs, size_t nbytes, uint64_t alignment, bool isVtcm);

  /// Frees the allocated memory region.
  void Free(void *ptr);

  /// Copies the data from source src into destination dst.
  void Copy(void *dst, void *src, size_t nbytes);

private:
  /// Manages runtime HexagonBuffer allocations
  std::unique_ptr<BufferManager> bufferManager;

  /// VTCM memory manager
  std::unique_ptr<VtcmPool> runtimeVtcm;

  /// HMX Context
  uint32_t hmx_context_id = 0;

  /// Power on and lock HMX context
  uint32_t initialize_and_acquire_hmx();

  void release_hmx() {
    if (hmx_context_id) {
      HAP_compute_res_hmx_unlock(hmx_context_id);
      HAP_compute_res_release(hmx_context_id);
      hmx_context_id = 0;
    }
  }
};
#endif // HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_
