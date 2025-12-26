//===- VTCMPool.h - VTCM pool manager  ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H

#include "HAP_compute_res.h"
#include "HexagonCommon.h"
#include <vector>

class VtcmPool {
public:
  /// Allocates all of VTCM memory, and manages from the runtime
  VtcmPool();

  /// Destruction deallocates the underlying VTCM allocation.
  ~VtcmPool();

  /// Prevent copy construction of VtcmPool.
  VtcmPool(const VtcmPool &) = delete;

  /// Prevent copy assignment with VtcmPool.
  VtcmPool &operator=(const VtcmPool &) = delete;

  /// Prevent move construction.
  VtcmPool(VtcmPool &&) = delete;

  /// Prevent move assignment.
  VtcmPool &operator=(VtcmPool &&) = delete;

  /// Allocate memory from the VTCM manager
  void *Allocate(size_t nbytes);

  /// Free nbytes from the allocated ptr
  void Free(void *ptr, size_t nbytes);

  /// Returns the total number of bytes in this pool
  size_t VtcmDeviceBytes() { return reinterpret_cast<size_t>(vtcmDeviceSize_); }

  /// Returns the total number of allocated bytes in this pool
  size_t VtcmAllocatedBytes() {
    return reinterpret_cast<size_t>(vtcmAllocatedSize_);
  }

  bool IsVtcm(void *ptr, unsigned size) {
    auto char_ptr = static_cast<char *>(ptr);
    CHECK(char_ptr != nullptr);
    auto char_vtcm = static_cast<char *>(vtcmData_);
    CHECK(vtcmData_ != nullptr);

    if (char_ptr >= char_vtcm &&
        (char_ptr + size) <= (char_vtcm + vtcmAllocatedSize_)) {
      return true;
    }
    return false;
  }

private:
  /// Total size of VTCM memory on device
  unsigned int vtcmDeviceSize_;

  /// Total size of VTCM pool allocated on device
  unsigned int vtcmAllocatedSize_;

  /// Pointer to the beginning of the pool
  void *vtcmData_;

  /// Context for HAP_compute_res_*
  unsigned int contextId_{0};

  /// List of allocations
  std::vector<std::pair<char *, size_t>> allocations_;

  /// List of free segments
  std::vector<std::pair<char *, size_t>> free_;

  /// Debug only dump of the state of the lists
  void DebugDump();
};

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H
