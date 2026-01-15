//===- HexagonBuffer.cpp                              ---------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <string>
#include <utility>

#include "HexagonAPI.h"
#include "HexagonBuffer.h"
#include "HexagonCommon.h"
#include "qurt_memory.h"

struct Allocation {
  Allocation(size_t allocatedBytes, size_t alignment)
      : allocatedBytes_(allocatedBytes), alignment_(alignment) {}
  virtual ~Allocation() {}
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;
  Allocation(Allocation &&) = delete;
  Allocation &operator=(Allocation &&) = delete;

  void *data_{nullptr};
  size_t allocatedBytes_;
  size_t alignment_;
};

struct DDRAllocation : public Allocation {
  DDRAllocation(size_t nbytes, size_t alignment)
      : Allocation(nbytes, alignment) {
    int ret = posix_memalign(&data_, alignment, nbytes);
    CHECK_EQ(ret, 0);
  }
  ~DDRAllocation() { free(data_); }
};

struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t nbytes, size_t alignment)
      : Allocation(nbytes, alignment) {
    // For simplicity, the current VTCM dynamic pool supports the following
    // alignments: less than
    // or equal to 128 (0x80), and 2k (0x800)
    CHECK(((alignment <= 0x80) || (alignment == 0x800)),
          "VTCMAllocation called for invalid alignment " << alignment);

    if (alignment == 0x800) {
      // Adjust size to be a multiple of 2k so that we will allocate from the
      // front of the pool.
      nbytes = (nbytes + 0x7ff) & -0x800;
    } else if (alignment <= 0x80) {
      // Adjust size to be a multiple of 128 so that we will allocate from the
      // back of the pool in 128 byte increments.
      nbytes = (nbytes + 0x7f) & -0x80;
    }
    if (allocatedBytes_ != nbytes) {
      allocatedBytes_ = nbytes;
    }
    CHECK(HexagonAPI::Global()->getVtcmPool(), "VTCM not acquired\n");
    data_ = HexagonAPI::Global()->getVtcmPool()->Allocate(allocatedBytes_);
  }
  ~VTCMAllocation() {
    HexagonAPI::Global()->getVtcmPool()->Free(data_, allocatedBytes_);
    data_ = nullptr;
  }
};

template <HexagonBuffer::StorageScope S>
std::unique_ptr<Allocation> Allocator(size_t nbytes, size_t alignment);

template <>
std::unique_ptr<Allocation>
Allocator<HexagonBuffer::StorageScope::kDDR>(size_t nbytes, size_t alignment) {
  return std::make_unique<DDRAllocation>(nbytes, alignment);
}

template <>
std::unique_ptr<Allocation>
Allocator<HexagonBuffer::StorageScope::kVTCM>(size_t nbytes, size_t alignment) {
  return std::make_unique<VTCMAllocation>(nbytes, alignment);
}

HexagonBuffer::HexagonBuffer(size_t nbytes, size_t alignment, bool isVtcm)
    : ndim_(1), nbytesPerAllocation_(nbytes) {
  SetStorageScope(isVtcm);

  std::unique_ptr<Allocation> alloca = nullptr;
  if (GetStorageScope() == StorageScope::kDDR) {
    alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
  } else if (GetStorageScope() == StorageScope::kVTCM) {
    alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
  }
  CHECK((alloca != nullptr), "could not create allocation");
  allocations_.push_back(alloca->data_);
  managedAllocations_.push_back(std::move(alloca));
}

HexagonBuffer::HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment,
                             bool isVtcm)
    : ndim_(2), nbytesPerAllocation_(nbytes) {
  SetStorageScope(isVtcm);

  size_t nbytesAligned = ((nbytes + (alignment - 1)) / alignment) * alignment;
  size_t nbytesMonolithic = nallocs * nbytesAligned;

  std::unique_ptr<Allocation> alloca = nullptr;
  if (GetStorageScope() == StorageScope::kDDR) {
    alloca = Allocator<StorageScope::kDDR>(nbytesMonolithic, alignment);
  } else if (GetStorageScope() == StorageScope::kVTCM) {
    alloca = Allocator<StorageScope::kVTCM>(nbytesMonolithic, alignment);
  }
  CHECK(alloca, "could not create allocation");

  for (size_t i = 0; i < nallocs; ++i) {
    void *alloc_offset =
        static_cast<unsigned char *>(alloca->data_) + i * nbytesAligned;
    allocations_.push_back(alloc_offset);
  }

  managedAllocations_.push_back(std::move(alloca));
}

HexagonBuffer::~HexagonBuffer() { managedAllocations_.clear(); }

void *HexagonBuffer::GetPointer() {
  CHECK((allocations_.size()), "Internal failure, allocations_ should be set "
                               "in HexagonBuffer constructor");
  CHECK(((ndim_ == 1) || (ndim_ == 2)),
        "HexagonBuffer should be either 1-d or 2-d");
  if (ndim_ == 1) {
    CHECK_EQ(allocations_.size(), 1);
    return allocations_[0];
  }
  // ndim_ == 2
  return allocations_.data();
}

HexagonBuffer::StorageScope HexagonBuffer::GetStorageScope() const {
  return storageScope_;
}

void HexagonBuffer::SetStorageScope(bool isVtcm) {
  if (isVtcm) {
    storageScope_ = StorageScope::kVTCM;
  } else {
    storageScope_ = StorageScope::kDDR;
  }
}

std::vector<MemoryCopy> BufferSet::MemoryCopies(const BufferSet &dest,
                                                const BufferSet &src,
                                                size_t bytesToCopy) {
  CHECK(bytesToCopy <= src.TotalBytes());
  CHECK(bytesToCopy <= dest.TotalBytes());

  auto pointer_to = [](const BufferSet &buf, size_t regionI,
                       size_t byteI) -> void * {
    void *region = buf.buffers[regionI];
    return static_cast<unsigned char *>(region) + byteI;
  };

  size_t num_srcRegions =
      (bytesToCopy + src.regionSizeBytes - 1) / src.regionSizeBytes;

  // First, determine all copies that do not cross boundaries in
  // either source or destination region.  This requires two loops, as
  // a single source region may overlap one or more destination
  // regions, and vice versa.
  std::vector<MemoryCopy> microCopies;
  for (size_t srcI = 0; srcI < num_srcRegions; srcI++) {
    size_t srcRegionBegin = srcI * src.regionSizeBytes;
    size_t srcRegionEnd =
        std::min((srcI + 1) * src.regionSizeBytes, bytesToCopy);

    size_t destIbegin = srcRegionBegin / dest.regionSizeBytes;
    size_t destIend = (srcRegionEnd - 1) / dest.regionSizeBytes + 1;
    for (size_t destI = destIbegin; destI < destIend; destI++) {
      size_t offsetBegin =
          std::max(srcRegionBegin, destI * dest.regionSizeBytes);
      size_t offset_end =
          std::min(srcRegionEnd, (destI + 1) * dest.regionSizeBytes);

      size_t numBytes = offset_end - offsetBegin;
      void *src_ptr = pointer_to(src, srcI, offsetBegin % src.regionSizeBytes);
      void *dest_ptr =
          pointer_to(dest, destI, offsetBegin % dest.regionSizeBytes);
      microCopies.push_back(MemoryCopy(dest_ptr, src_ptr, numBytes));
    }
  }

  return microCopies;
}

std::vector<MemoryCopy>
MemoryCopy::MergeAdjacent(std::vector<MemoryCopy> microCopies) {
  std::sort(
      microCopies.begin(), microCopies.end(),
      [](const MemoryCopy &a, const MemoryCopy &b) { return a.src < b.src; });

  std::vector<MemoryCopy> macroCopies;
  for (const auto &copy : microCopies) {
    if (macroCopies.size() && macroCopies.back().IsDirectlyBefore(copy)) {
      macroCopies.back().numBytes += copy.numBytes;
    } else {
      macroCopies.push_back(copy);
    }
  }
  return macroCopies;
}

void hexagonBufferCopyAcrossRegions(const BufferSet &dest, const BufferSet &src,
                                    size_t bytesToCopy, bool srcIsHexbuff,
                                    bool destIsHexbuff) {
  // First, determine all copies that do not cross boundaries in
  // either source or destination region.
  // auto microCopies = BufferSet::MemoryCopies(dest, src, bytesToCopy);
  auto microCopies = BufferSet::MemoryCopies(dest, src, bytesToCopy);

  // If regions are contiguously allocated, we can reduce the number
  // of copies required by merging adjacent copies.
  auto macroCopies = MemoryCopy::MergeAdjacent(std::move(microCopies));

  // Finally, do the memory copies.
  for (const auto &copy : macroCopies) {
    // if src is a HexagonBuffer, invalidate it before the memcpy
    if (srcIsHexbuff) {
      qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(copy.src),
                           copy.numBytes, QURT_MEM_CACHE_INVALIDATE,
                           QURT_MEM_DCACHE);
    }

    // TODO(HWE): Switch to ION Buffer to avoid need for memcpy and potentially
    // lighten or alleviate the burden of cache invalidation in this code
    memcpy(copy.dest, copy.src, copy.numBytes);

    // if dest is a HexagonBuffer, flush it after the memcpy
    if (destIsHexbuff) {
      qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(copy.dest),
                           copy.numBytes, QURT_MEM_CACHE_FLUSH,
                           QURT_MEM_DCACHE);
    }
  }
}

void HexagonBuffer::CopyTo(void *data, size_t nbytes) const {
  BufferSet src(allocations_.data(), allocations_.size(), nbytesPerAllocation_);
  BufferSet dest(&data, 1, nbytes);

  hexagonBufferCopyAcrossRegions(dest, src, nbytes, true /* srcIsHexbuff */,
                                 false /* destIsHexbuff */);
}

void HexagonBuffer::CopyFrom(void *data, size_t nbytes) {
  BufferSet src(&data, 1, nbytes);
  BufferSet dest(allocations_.data(), allocations_.size(),
                 nbytesPerAllocation_);

  hexagonBufferCopyAcrossRegions(dest, src, nbytes, false /* srcIsHexbuff */,
                                 true /* destIsHexbuff */);
}

void HexagonBuffer::CopyFrom(const HexagonBuffer &other, size_t nbytes) {
  BufferSet src(other.allocations_.data(), other.allocations_.size(),
                other.nbytesPerAllocation_);
  BufferSet dest(allocations_.data(), allocations_.size(),
                 nbytesPerAllocation_);

  hexagonBufferCopyAcrossRegions(dest, src, nbytes, true /* srcIsHexbuff */,
                                 true /* destIsHexbuff */);
}
