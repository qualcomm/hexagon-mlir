//===- VTCMPool.cpp - VTCM pool                                    --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "VTCMPool.h"
#include "HAP_compute_res.h"
#include "HexagonCommon.h"

VtcmPool::VtcmPool() {
  compute_res_attr_t resInfo;
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&resInfo));

  unsigned int availBlockSize;
  compute_res_vtcm_page_t totalBlockLayout;
  compute_res_vtcm_page_t availBlockLayout;
  unsigned int pageSize = 0;

  HEXAGON_SAFE_CALL(compute_resource_query_VTCM(
      /* application_id = */ 0, &vtcmDeviceSize_, &totalBlockLayout,
      &availBlockSize, &availBlockLayout));

  // Use the max page size form the page_list
  for (int i = 0; i < totalBlockLayout.page_list_len; i++) {
    if (pageSize < totalBlockLayout.page_list[i].page_size) {
      pageSize = totalBlockLayout.page_list[i].page_size;
    }
  }

  CHECK((availBlockSize >= (1024 * 1024)), "Less than 1MB VTCM available");

  // allocate nbytes of vtcm on a single page
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param_v2(
      &resInfo,
      /*vtcm_size = */ vtcmDeviceSize_,
      /*min_page_size = */ pageSize,
      /*min_vtcm_size = */ availBlockSize));

  // TODO(HWE): Investigate why a non-zero timeout results in hanging, both in
  // the simulator and on hardware.
  contextId_ = HAP_compute_res_acquire(&resInfo, /*timeout = */ 0);
  CHECK((contextId_),
        "HAP_compute_res_acquire failed to acquire requested VTCM resource");
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_get_vtcm_ptr_v2(&resInfo, &vtcmData_,
                                                         &vtcmAllocatedSize_));
  CHECK((vtcmData_ != nullptr),
        "HAP_compute_res_acquire returned nullptr when allocating VTCM");
  CHECK((vtcmAllocatedSize_ >= availBlockSize),
        "HAP_compute_res_acquire failed to allocate minimum amount of VTCM");
  free_.emplace_back(std::pair<char *, size_t>(static_cast<char *>(vtcmData_),
                                               vtcmAllocatedSize_));
}

VtcmPool::~VtcmPool() {
  HEXAGON_SAFE_CALL(HAP_compute_res_release(contextId_));
}

void *VtcmPool::Allocate(size_t nbytes) {
  CHECK((!free_.empty()), "No free VTCM");
  CHECK((nbytes >= 0x80),
        "Minimum VTCM alloation must be 128 bytes - nbytes " << nbytes);

  // If not aligned on a 2k block, allocate from the end to avoid fragmentation
  if (nbytes & size_t(0x7FF)) {
    auto lastFreeEntry = free_.end();
    lastFreeEntry--;
    CHECK((lastFreeEntry->second >= nbytes),
          "Not enough contiguous VTCM space at the end to allocate");
    char *ptr = lastFreeEntry->first + (lastFreeEntry->second - nbytes);
    allocations_.emplace_back(std::pair<char *, size_t>(ptr, nbytes));
    lastFreeEntry->second -= nbytes;
    if (lastFreeEntry->second == 0) {
      free_.erase(lastFreeEntry);
    }
    return ptr;
  }

  auto entryToAllocate = free_.begin();
  for (auto it = free_.begin(); it != free_.end(); it++) {
    if ((entryToAllocate->second < nbytes ||
         it->second < entryToAllocate->second) &&
        it->second >= nbytes) {
      entryToAllocate = it;
      if (entryToAllocate->second == nbytes) {
        break;
      }
    }
  }
  CHECK((entryToAllocate->second >= nbytes),
        "Not enough contiguous VTCM space to allocate");
  char *ptr = entryToAllocate->first;
  allocations_.emplace(allocations_.end(),
                       std::pair<char *, size_t>(ptr, nbytes));

  if (entryToAllocate->second == nbytes) {
    free_.erase(entryToAllocate);
  } else {
    entryToAllocate->first = entryToAllocate->first + nbytes;
    entryToAllocate->second = entryToAllocate->second - nbytes;
  }
  return ptr;
}

void VtcmPool::Free(void *ptr, size_t nbytes) {
  char *ptrToFree = static_cast<char *>(ptr);
  auto it = std::find_if(allocations_.begin(), allocations_.end(),
                         [&](auto entry) { return entry.first == ptrToFree; });
  CHECK((it != allocations_.end()),
        "Attempted to free a pointer that had not been allocated");
  CHECK((it->second == nbytes),
        "Attempted to free a different size than was allocated");
  allocations_.erase(it);

  it = std::lower_bound(free_.begin(), free_.end(),
                        std::pair<char *, size_t>(ptrToFree, nbytes),
                        [](auto p, auto q) { return p.first <= q.first; });
  if (it == free_.end()) {
    // Insert an entry at the end
    it = free_.emplace(it, std::pair<char *, size_t>(ptrToFree, nbytes));
  } else {
    CHECK((ptrToFree != it->first),
          "Attempting to free a pointer that was already free");
    CHECK((ptrToFree + nbytes <= it->first),
          "free_ is in an inconsistent state, freed block overlaps with next");
    if (ptrToFree + nbytes == it->first) {
      // Make this entry bigger
      it->first = ptrToFree;
      it->second += nbytes;
    } else {
      // Insert an entry before this
      it = free_.emplace(it, std::pair<char *, size_t>(ptrToFree, nbytes));
    }
  }

  // Check for overlap with the previous entry
  if (it != free_.begin()) {
    auto it_prev = it;
    it_prev--;
    CHECK((it_prev->first + it_prev->second <= ptrToFree),
          "free_ is in an inconsistent state, freed block overlaps with "
          "previous");
    if (it_prev->first + it_prev->second == ptrToFree) {
      it_prev->second += it->second;
      free_.erase(it);
    }
  }
}
