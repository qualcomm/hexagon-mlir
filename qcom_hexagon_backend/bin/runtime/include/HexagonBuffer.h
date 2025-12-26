//===- HexagonBuffer.h  ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Structure used to track/coalesce memory copies
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INC_HEXAGONBUFFER_H_
#define HEXAGON_BIN_RUNTIME_INC_HEXAGONBUFFER_H_

#include <memory>
#include <vector>

struct Allocation;

class HexagonBuffer {
public:
  /// Allocate 1d (contiguous) memory within memory scopes.
  HexagonBuffer(size_t nbytes, size_t alignment, bool isVtcm);

  /// Allocate 2d (discontiguous) memory within memory scopes.
  HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment, bool isVtcm);

  /// Destruction deallocates the underlying allocations.
  ~HexagonBuffer();

  /// Prevent copy construction of HexagonBuffers.
  HexagonBuffer(const HexagonBuffer &) = delete;

  /// Prevent copy assignment with HexagonBuffers.
  HexagonBuffer &operator=(const HexagonBuffer &) = delete;

  /// Prevent move construction.
  HexagonBuffer(HexagonBuffer &&) = delete;

  /// Prevent move assignment.
  HexagonBuffer &operator=(HexagonBuffer &&) = delete;

  // Return data pointer into the buffer
  // The returned pointer is intended for use as the runtime value
  // corresponding to the `Var BufferNode::data` of a buffer.  The
  // return type depends on the dimensionality of the buffer being
  // accessed, and must be compatible with the usage defined in
  // `CodeGenHexagon::CreateBufferPtr`.
  //
  // For a 1-d buffer, this pointer can be cast to a `T*` and accessed
  // as a 1-d array (e.g. `static_cast<int32_t*>(GetPointer())[i]`).
  // For a 2-d buffer, this pointer can be cast to a `T**` and accessed
  // as a 2-d array (e.g. `static_cast<int32_t**>(GetPointer())[i][j]`).
  void *GetPointer();

  /// Memory scopes managed by a Hexagon Buffer.
  enum class StorageScope {
    // System DDR corresponding to global storage.
    kDDR,
    // VTCM memory corresponding to global.vtcm storage.
    kVTCM,
  };

  /// Return storage scope of underlying allocation.
  StorageScope GetStorageScope() const;

  /// Copy data from a Hexagon Buffer an external buffer.
  void CopyTo(void *data, size_t nbytes) const;

  /// Copy data from an external buffer to a Hexagon Buffer.
  void CopyFrom(void *data, size_t nbytes);

  /// Copy data from one Hexagon Buffer to another.
  void CopyFrom(const HexagonBuffer &other, size_t nbytes);

  /// Return the total number of bytes in this buffer
  size_t TotalBytes() const {
    return nbytesPerAllocation_ * allocations_.size();
  }

  size_t GetNdim() const { return ndim_; }

private:
  /// Assign a storage scope to the buffer.
  void SetStorageScope(bool isVtcm);
  /// Array of raw pointer allocations required by the buffer.
  // For 1d (contiguous) storage a single allocation will result.
  // For 2d (discontiguous) storage `nallocs` allocations will result.
  std::vector<void *> allocations_;
  /// Managed allocations follow RAII and are released at destruction.
  std::vector<std::unique_ptr<Allocation>> managedAllocations_;
  /// The underlying storage type in which the allocation resides.
  size_t ndim_;
  size_t nbytesPerAllocation_;
  StorageScope storageScope_;
};

/// Structure used to track/coalesce memory copies
struct MemoryCopy {
  static std::vector<MemoryCopy>
  MergeAdjacent(std::vector<MemoryCopy> microCopies);

  MemoryCopy(void *dest, void *src, size_t numBytes)
      : dest(dest), src(src), numBytes(numBytes) {}

  bool IsDirectlyBefore(const MemoryCopy &other) {
    void *src_end = static_cast<unsigned char *>(src) + numBytes;
    void *dest_end = static_cast<unsigned char *>(dest) + numBytes;
    return (src_end == other.src) && (dest_end == other.dest);
  }

  void *dest;
  void *src;
  size_t numBytes;
};

struct BufferSet {
  // Determine all copies that do not cross boundaries in either
  // source or destination region.
  static std::vector<MemoryCopy>
  MemoryCopies(const BufferSet &dest, const BufferSet &src, size_t bytesToCopy);

  BufferSet(void *const *buffers, size_t numRegions, size_t regionSizeBytes)
      : buffers(buffers), numRegions(numRegions),
        regionSizeBytes(regionSizeBytes) {}

  size_t TotalBytes() const { return numRegions * regionSizeBytes; }

  void *const *buffers;
  size_t numRegions;
  size_t regionSizeBytes;
};

void hexagonBufferCopyAcrossRegions(const BufferSet &dest, const BufferSet &src,
                                    size_t bytesToCopy, bool srcIsHexbuff,
                                    bool destIsHexbuff);

#endif // HEXAGON_BIN_RUNTIME_INC_HEXAGONBUFFER_H_
