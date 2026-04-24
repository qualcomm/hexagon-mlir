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

#if defined(__hexagon__)
#include <cstdio>
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
// Constants
constexpr size_t kSmallAlignment = 128;
constexpr size_t kLargeAlignment = 2048;
constexpr size_t kLargeThreshold = 2048;

// Environment variable check (once at startup)
static int getVTCMDebugLevel() {
  static int level = -1;
  if (level == -1) {
    const char *env = std::getenv("VTCM_DEBUG");
    level = env ? std::atoi(env) : 0;
  }
  return level;
}

// Debug logging wrapper
template <typename... Args> void vtcmDebugLog(int minLevel, Args &&...args) {
  if (getVTCMDebugLevel() >= minLevel) {
    (std::cout << ... << std::forward<Args>(args)) << std::endl;
  }
}

// Size formatting helpers
const char *fmtKB(size_t bytes) {
  static thread_local char buf[32];
  snprintf(buf, sizeof(buf), "%.1fKB", bytes / 1024.0);
  return buf;
}

const char *fmtMB(size_t bytes) {
  static thread_local char buf[32];
  snprintf(buf, sizeof(buf), "%.2fMB", bytes / (1024.0 * 1024.0));
  return buf;
}

const char *fmtPct(size_t part, size_t total) {
  static thread_local char buf[32];
  if (total == 0) {
    snprintf(buf, sizeof(buf), "0.0%%");
  } else {
    snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * part / total);
  }
  return buf;
}

// Alignment calculation (shared by Allocate and Free)
size_t alignSize(size_t nbytes) {
  size_t alignment =
      (nbytes >= kLargeThreshold) ? kLargeAlignment : kSmallAlignment;
  return (nbytes + (alignment - 1)) & ~(alignment - 1);
}

// Find the last free block (by address)
std::vector<std::pair<char *, size_t>>::iterator
findLastFreeBlock(std::vector<std::pair<char *, size_t>> &free) {
  if (free.empty())
    return free.end();
  auto last = free.end();
  last--;
  return last;
}

// Find best-fit block (smallest sufficient)
std::vector<std::pair<char *, size_t>>::iterator
findBestFit(std::vector<std::pair<char *, size_t>> &free, size_t nbytes) {
  auto bestFit = free.end();
  for (auto it = free.begin(); it != free.end(); it++) {
    if (it->second >= nbytes) {
      if (bestFit == free.end() || it->second < bestFit->second) {
        bestFit = it;
        if (bestFit->second == nbytes) {
          break; // Perfect fit
        }
      }
    }
  }
  return bestFit;
}

// Calculate total allocated size
size_t
calcTotalAllocated(const std::vector<std::pair<char *, size_t>> &allocations) {
  size_t total = 0;
  for (const auto &alloc : allocations) {
    total += alloc.second;
  }
  return total;
}

// Find largest free block size
size_t
calcLargestFreeBlock(const std::vector<std::pair<char *, size_t>> &free) {
  size_t largest = 0;
  for (const auto &block : free) {
    if (block.second > largest) {
      largest = block.second;
    }
  }
  return largest;
}

// Memory map visualization
void printMemoryMap(const std::vector<std::pair<char *, size_t>> &allocations,
                    const std::vector<std::pair<char *, size_t>> &free,
                    void *baseAddr, size_t totalSize) {
  if (getVTCMDebugLevel() < 2)
    return;

  struct Block {
    char *addr;
    size_t size;
    bool isAlloc;
    bool operator<(const Block &other) const { return addr < other.addr; }
  };

  std::vector<Block> blocks;
  for (const auto &a : allocations) {
    blocks.push_back({a.first, a.second, true});
  }
  for (const auto &f : free) {
    blocks.push_back({f.first, f.second, false});
  }

  std::sort(blocks.begin(), blocks.end());

  size_t totalAlloc = 0;
  for (const auto &b : blocks) {
    if (b.isAlloc)
      totalAlloc += b.size;
  }

  std::cout << "[VTCM] Memory Map (used=" << fmtPct(totalAlloc, totalSize)
            << ", " << allocations.size() << " allocs, " << free.size()
            << " free blocks):" << std::endl;

  char *base = static_cast<char *>(baseAddr);
  for (const auto &block : blocks) {
    size_t startOffset = block.addr - base;
    size_t endOffset = startOffset + block.size;

    std::cout << "  [" << (block.isAlloc ? "ALLOC" : "FREE ") << "] "
              << "0x" << std::hex << std::setfill('0') << std::setw(8)
              << startOffset << "-0x" << std::setw(8) << endOffset << std::dec
              << " (" << (block.isAlloc ? fmtKB(block.size) : fmtMB(block.size))
              << ")" << std::endl;
  }
}
} // namespace

// Allocation strategy:
// - Small allocations (< 2KB): 128-byte aligned, allocated from end of last
//   free block (by address) to reduce fragmentation in the main pool area.
//   Falls back to best-fit if end allocation fails.
// - Large allocations (>= 2KB): 2KB aligned (hardware requirement for certain
//   tensor units), use best-fit to minimize fragmentation.
//
// The previous strategy allocated unaligned blocks from the end, which created
// unusable fragments when freed. By maintaining alignment, we avoid this bug
// while preserving memory efficiency (0% overhead for 128-byte allocations).
//
// Debug logging:
// - VTCM_DEBUG=0 (default): No debug output
// - VTCM_DEBUG=1: Concise operation summaries
// - VTCM_DEBUG=2: Detailed memory maps after each operation
// - Errors always print memory map (even without VTCM_DEBUG)
//
// Example usage:
//   VtcmPool pool;
//   void* small = pool.Allocate(256);  // 128-byte aligned
//   void* large = pool.Allocate(4096); // 2KB aligned
//   pool.Free(small, 256);
//   pool.Free(large, 4096);

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

  // Use the max page size from the page_list
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

  vtcmAllocatedPtr_ = vtcmData_;

  vtcmDebugLog(1, "[VTCM] Init: device=", fmtMB(vtcmDeviceSize_),
               " allocated=", fmtMB(vtcmAllocatedSize_));

  HEXAGON_PRINT(HIGH,
                "[hexagon-mlir runtime] VTCM: device=%u bytes, available=%u "
                "bytes, allocated=%u bytes\n",
                vtcmDeviceSize_, availBlockSize, vtcmAllocatedSize_);

#if defined(__hexagon__)
  // Also print to stdout so it can be captured by run wrappers.
  std::printf("[hexagon-mlir runtime] VTCM: device=%u bytes, available=%u "
              "bytes, allocated=%u bytes\n",
              vtcmDeviceSize_, availBlockSize, vtcmAllocatedSize_);
#endif

  free_.emplace_back(std::pair<char *, size_t>(static_cast<char *>(vtcmData_),
                                               vtcmAllocatedSize_));

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

VtcmPool::~VtcmPool() {
  // Check memory accounting
  assert(checkMemoryAccountedFor());

  // Leak detection
  if (!allocations_.empty()) {
    size_t totalLeaked = calcTotalAllocated(allocations_);

    std::cerr << "[VTCM] WARNING: Leaked " << fmtKB(totalLeaked) << " in "
              << allocations_.size() << " allocations" << std::endl;

    printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
  }

  vtcmDebugLog(1, "[VTCM] Cleanup: released ", fmtMB(vtcmAllocatedSize_));

  HEXAGON_SAFE_CALL(HAP_compute_res_release(contextId_));
}

void *VtcmPool::Allocate(size_t nbytes) {
  // Edge case: zero-size allocation
  if (nbytes == 0) {
    vtcmDebugLog(1, "[VTCM] WARNING: Zero-size allocation requested");
    return nullptr;
  }

  size_t original_nbytes = nbytes;
  nbytes = alignSize(nbytes);

  // Edge case: allocation larger than total pool
  if (nbytes > vtcmAllocatedSize_) {
    std::cerr << "[VTCM] ERROR: Requested " << fmtKB(nbytes)
              << " exceeds total pool size " << fmtMB(vtcmAllocatedSize_)
              << std::endl;
    return nullptr;
  }

  vtcmDebugLog(2, "[VTCM] Alloc request: ", fmtKB(original_nbytes),
               " → aligned: ", fmtKB(nbytes));

  char *ptr = nullptr;

  // Small allocation: try END of last free block
  if (nbytes < kLargeThreshold) {
    ptr = tryAllocateFromEnd(nbytes);
    if (ptr != nullptr) {
      vtcmDebugLog(2, "[VTCM]   Allocated from end of last block");
    } else {
      vtcmDebugLog(2, "[VTCM]   End allocation failed, using best-fit");
    }
  }

  // Large allocation OR small allocation fallback: use best-fit
  if (ptr == nullptr) {
    ptr = allocateBestFit(nbytes);
  }

  if (ptr == nullptr) {
    handleAllocationFailure(nbytes);
    return nullptr;
  }

  logAllocationSuccess(ptr, nbytes);

#ifndef NDEBUG
  validateInvariants();
#endif

  return ptr;
}

// Try to allocate from end of last free block
char *VtcmPool::tryAllocateFromEnd(size_t nbytes) {
  auto lastBlock = findLastFreeBlock(free_);

  if (lastBlock == free_.end() || lastBlock->second < nbytes) {
    return nullptr;
  }

  // Allocate from end of last block
  char *ptr = lastBlock->first + (lastBlock->second - nbytes);
  allocations_.emplace_back(std::pair<char *, size_t>(ptr, nbytes));
  lastBlock->second -= nbytes;

  if (lastBlock->second == 0) {
    free_.erase(lastBlock);
  }

  return ptr;
}

// Allocate using best-fit strategy
char *VtcmPool::allocateBestFit(size_t nbytes) {
  auto bestFit = findBestFit(free_, nbytes);

  if (bestFit == free_.end()) {
    return nullptr;
  }

  // Allocate from beginning of best-fit block
  char *ptr = bestFit->first;
  allocations_.emplace_back(std::pair<char *, size_t>(ptr, nbytes));

  bestFit->first += nbytes;
  bestFit->second -= nbytes;

  if (bestFit->second == 0) {
    free_.erase(bestFit);
  }

  return ptr;
}

// Handle allocation failure
void VtcmPool::handleAllocationFailure(size_t nbytes) {
  size_t totalAllocated = calcTotalAllocated(allocations_);
  size_t largestFree = calcLargestFreeBlock(free_);
  float fragScore = getFragmentationScore();

  std::cerr << "[VTCM] ERROR: Allocation failed!" << std::endl;
  std::cerr << "  Requested: " << fmtKB(nbytes) << std::endl;
  std::cerr << "  Largest free block: " << fmtKB(largestFree) << std::endl;
  std::cerr << "  Total used: " << fmtPct(totalAllocated, vtcmAllocatedSize_)
            << " (" << fmtKB(totalAllocated) << ")" << std::endl;
  std::cerr << "  Free blocks: " << free_.size() << std::endl;
  std::cerr << "  Fragmentation: " << (int)(fragScore * 100) << "%"
            << std::endl;

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

// Log successful allocation
void VtcmPool::logAllocationSuccess(char *ptr, size_t nbytes) {
  size_t totalAllocated = calcTotalAllocated(allocations_);
  size_t offset = ptr - static_cast<char *>(vtcmAllocatedPtr_);

  vtcmDebugLog(1, "[VTCM] Alloc: ", fmtKB(nbytes), " @ 0x", std::hex, offset,
               std::dec, " | used=", fmtPct(totalAllocated, vtcmAllocatedSize_),
               " allocs=", allocations_.size(), " free_blocks=", free_.size());

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

void VtcmPool::Free(void *ptr, size_t nbytes) {
  // Edge case: null pointer
  if (ptr == nullptr) {
    vtcmDebugLog(1, "[VTCM] WARNING: Attempted to free null pointer");
    return;
  }

  // Edge case: zero size
  if (nbytes == 0) {
    vtcmDebugLog(1, "[VTCM] WARNING: Attempted to free zero-size allocation");
    return;
  }

  size_t original_nbytes = nbytes;
  nbytes = alignSize(nbytes);

  vtcmDebugLog(2, "[VTCM] Free request: ", fmtKB(original_nbytes),
               " → aligned: ", fmtKB(nbytes));

  // Validate and remove from allocations
  char *ptrToFree = static_cast<char *>(ptr);
  auto it = std::find_if(allocations_.begin(), allocations_.end(),
                         [&](auto entry) { return entry.first == ptrToFree; });

  CHECK((it != allocations_.end()),
        "Attempted to free a pointer that had not been allocated");
  CHECK((it->second == nbytes),
        "Attempted to free a different size than was allocated");

  allocations_.erase(it);

  // Coalesce and add to free list
  size_t numCoalesced = coalesceAndAddToFreeList(ptrToFree, nbytes);

  // Log the free operation
  logFreeSuccess(ptrToFree, nbytes, numCoalesced);

#ifndef NDEBUG
  validateInvariants();
#endif
}

// Coalesce adjacent free blocks and add to free list
size_t VtcmPool::coalesceAndAddToFreeList(char *ptr, size_t nbytes) {
  size_t numFreeBlocksBefore = free_.size();

  auto it = std::lower_bound(free_.begin(), free_.end(),
                             std::pair<char *, size_t>(ptr, nbytes),
                             [](auto p, auto q) { return p.first < q.first; });
  if (it == free_.end()) {
    // Insert an entry at the end
    it = free_.emplace(it, std::pair<char *, size_t>(ptr, nbytes));
  } else {
    CHECK((ptr != it->first),
          "Attempting to free a pointer that was already free");
    CHECK((ptr + nbytes <= it->first),
          "free_ is in an inconsistent state, freed block overlaps with next");
    if (ptr + nbytes == it->first) {
      // Make this entry bigger (coalesce with next)
      it->first = ptr;
      it->second += nbytes;
    } else {
      // Insert an entry before this
      it = free_.emplace(it, std::pair<char *, size_t>(ptr, nbytes));
    }
  }

  // Check for overlap with the previous entry
  if (it != free_.begin()) {
    auto it_prev = it;
    it_prev--;
    CHECK((it_prev->first + it_prev->second <= ptr),
          "free_ is in an inconsistent state, freed block overlaps with "
          "previous");
    if (it_prev->first + it_prev->second == ptr) {
      // Coalesce with previous
      it_prev->second += it->second;
      free_.erase(it);
    }
  }

  return numFreeBlocksBefore - free_.size() + 1;
}

// Log successful free operation
void VtcmPool::logFreeSuccess(char *ptr, size_t nbytes, size_t numCoalesced) {
  size_t offset = ptr - static_cast<char *>(vtcmAllocatedPtr_);

  vtcmDebugLog(1, "[VTCM] Free: ", fmtKB(nbytes), " @ 0x", std::hex, offset,
               std::dec, " | coalesced ", numCoalesced, " blocks");

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

// Query methods
size_t VtcmPool::getTotalAllocated() const {
  return calcTotalAllocated(allocations_);
}

size_t VtcmPool::getTotalFree() const {
  size_t total = 0;
  for (const auto &block : free_) {
    total += block.second;
  }
  return total;
}

size_t VtcmPool::getLargestFreeBlock() const {
  return calcLargestFreeBlock(free_);
}

float VtcmPool::getFragmentationScore() const {
  if (free_.empty())
    return 0.0f;

  size_t totalFree = getTotalFree();
  size_t largestFree = getLargestFreeBlock();

  if (totalFree == 0)
    return 0.0f;

  // Fragmentation = 1 - (largest_free / total_free)
  // 0 = one big block (no fragmentation)
  // 1 = many tiny blocks (high fragmentation)
  return 1.0f - (static_cast<float>(largestFree) / totalFree);
}

void VtcmPool::printState() const {
  std::cout << "VTCM Pool State:" << std::endl;
  std::cout << "  Total: " << fmtMB(vtcmAllocatedSize_) << std::endl;
  std::cout << "  Allocated: " << fmtKB(getTotalAllocated()) << " ("
            << fmtPct(getTotalAllocated(), vtcmAllocatedSize_) << ")"
            << std::endl;
  std::cout << "  Free: " << fmtKB(getTotalFree()) << std::endl;
  std::cout << "  Largest free: " << fmtKB(getLargestFreeBlock()) << std::endl;
  std::cout << "  Allocations: " << allocations_.size() << std::endl;
  std::cout << "  Free blocks: " << free_.size() << std::endl;
  std::cout << "  Fragmentation: " << (int)(getFragmentationScore() * 100)
            << "%" << std::endl;
}

// Validation methods (debug builds only)
void VtcmPool::validateInvariants() const {
#ifndef NDEBUG
  // Check that allocations don't overlap
  for (auto it1 = allocations_.begin(); it1 != allocations_.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != allocations_.end(); ++it2) {
      char *end1 = it1->first + it1->second;
      char *end2 = it2->first + it2->second;
      assert((end1 <= it2->first || end2 <= it1->first) &&
             "Allocations overlap!");
    }
  }

  // Check that free blocks don't overlap
  for (auto it1 = free_.begin(); it1 != free_.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != free_.end(); ++it2) {
      char *end1 = it1->first + it1->second;
      char *end2 = it2->first + it2->second;
      assert((end1 <= it2->first || end2 <= it1->first) &&
             "Free blocks overlap!");
    }
  }

  // Check that allocations and free blocks don't overlap
  for (const auto &alloc : allocations_) {
    for (const auto &freeBlock : free_) {
      char *allocEnd = alloc.first + alloc.second;
      char *freeEnd = freeBlock.first + freeBlock.second;
      assert((allocEnd <= freeBlock.first || freeEnd <= alloc.first) &&
             "Allocation and free block overlap!");
    }
  }
#endif
}

bool VtcmPool::checkMemoryAccountedFor() const {
  size_t totalAllocated = getTotalAllocated();
  size_t totalFree = getTotalFree();
  size_t accounted = totalAllocated + totalFree;

  if (accounted != vtcmAllocatedSize_) {
    std::cerr << "[VTCM] ERROR: Memory accounting mismatch!" << std::endl;
    std::cerr << "  Allocated: " << totalAllocated << std::endl;
    std::cerr << "  Free: " << totalFree << std::endl;
    std::cerr << "  Total: " << accounted << std::endl;
    std::cerr << "  Expected: " << vtcmAllocatedSize_ << std::endl;
    return false;
  }
  return true;
}

void VtcmPool::DebugDump() {
  printState();
  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}
