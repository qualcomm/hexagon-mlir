//===- BufferManager.h - hexagon buffer manager        --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef BUFFERMANAGER_H_
#define BUFFERMANAGER_H_

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HexagonBuffer.h"
#include "HexagonBufferAlias.h"
#include "HexagonCommon.h"

class HexagonBufferAlias;

/// Hashes a HexagonBuffer::CacheKey. Defined by hand (not std::hash) so the key
/// stays a plain aggregate owned by HexagonBuffer.
struct HexagonBufferCacheKeyHash {
  size_t operator()(const HexagonBuffer::CacheKey &key) const {
    size_t h = std::hash<size_t>{}(key.numAllocations);
    auto mix = [&h](size_t value) {
      h ^= value + 0x9e3779b9u + (h << 6) + (h >> 2);
    };
    mix(std::hash<size_t>{}(key.bytesPerAllocation));
    mix(std::hash<size_t>{}(key.alignment));
    mix(std::hash<bool>{}(key.isVtcm));
    return h;
  }
};

class BufferManager {
public:
  ~BufferManager() {
    if (!bufferMap_.empty()) {
      CHECK((true), "BufferManager is not empty upon destruction");
    }
  }

  /// Free a HexagonBuffer.
  ///
  /// The wrapper (HexagonBuffer + its Allocation and vectors) and its reserved
  /// storage are kept in a free cache keyed by footprint, so the next launch's
  /// same-shaped allocation reuses them instead of rebuilding the object and
  /// re-running the pool's best-fit/coalesce bookkeeping. The pointer is
  /// *removed* from bufferMap_ either way, so a use-after-free still fails the
  /// same CHECK/Copy lookups as before; the cached buffer is simply not "live".
  void FreeHexagonBuffer(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bufferMap_.find(ptr);
    CHECK((it != bufferMap_.end()),
          "Attempt made to free unknown or already freed allocation");
    CHECK(it->second != nullptr);
    std::unique_ptr<HexagonBuffer> buf = std::move(it->second);
    bufferMap_.erase(it);

    size_t bytes = buf->GetAllocatedBytes();
    if (bytes != 0 && bytes <= kMaxCachedBytes &&
        cachedBytes_ + bytes <= kMaxCachedBytes) {
      cachedBytes_ += bytes;
      freeCache_[buf->GetCacheKey()].push_back(std::move(buf));
    }
    // Otherwise `buf` is destroyed at the end of this scope and the block
    // returns to the pool (the bounded-cache fallback).
  }

  /// Allocate a HexagonBuffer.
  template <typename... Args> void *AllocateHexagonBuffer(Args &&...args) {
    std::lock_guard<std::mutex> lock(mutex_);

    HexagonBuffer::CacheKey key = MakeCacheKey(args...);
    auto cached = freeCache_.find(key);
    if (cached != freeCache_.end() && !cached->second.empty()) {
      std::unique_ptr<HexagonBuffer> buf = std::move(cached->second.back());
      cached->second.pop_back();
      cachedBytes_ -= buf->GetAllocatedBytes();
      void *ptr = buf->GetPointer();
      bufferMap_.insert({ptr, std::move(buf)});
      return ptr;
    }

    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    if (!buf->HasValidAllocation()) {
      // The pool could not satisfy the request while the cache was holding
      // storage: drop the cache (returning those blocks to the pool) and run the
      // real allocator once more. The cache is a perf aid, never a reason to
      // fail an allocation that would otherwise succeed.
      freeCache_.clear();
      cachedBytes_ = 0;
      buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    }
    void *ptr = buf->GetPointer();
    bufferMap_.insert({ptr, std::move(buf)});
    return ptr;
  }

  /// Returns a pointer to crouton-table that is constructed using `nBytes`
  /// sized `buffer`. The table size is expected to be `nBytes/CROUTON_SIZE`
  void *CreateBufferAlias(void *ptr, size_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    HexagonBuffer *buffer = FindBuffer<HexagonBuffer>(ptr, bufferMap_);
    auto bufferAlias =
        std::make_unique<hexagon::HexagonBufferAlias>(*buffer, nbytes);

    auto *returnPtr = bufferAlias->GetCroutonTableBase();
    // bufferAliasMap_ would be needed to find aliases given a base pointer when
    // using HexagonBuffer::Copy calls
    // TODO: Modify HexagonBuffer::Copy to support aliases
    bufferAliasMap_.insert({returnPtr, std::move(bufferAlias)});
    return returnPtr;
  }

  /// Takes the crouton pointer table and returns the base pointer to the
  /// contiguous memref underneath
  void *GetOrigBufferFromAlias(void *croutonTablePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    hexagon::HexagonBufferAlias *aliasPtr =
        FindBuffer<hexagon::HexagonBufferAlias>(croutonTablePtr,
                                                bufferAliasMap_);
    assert(aliasPtr != nullptr &&
           "Expected the ptr to be a valid buffer alias created by "
           "memref_to_crouton op");
    auto *buffer = aliasPtr->origBuffer;
    return buffer->GetPointer();
  }

  /// Finds and returns the pointer from the given map if it exists and a
  /// nullptr otherwise
  template <typename BufferType>
  BufferType *
  FindBuffer(void *ptr,
             std::unordered_map<void *, std::unique_ptr<BufferType>> &map_) {
    auto it = map_.find(ptr);
    if (it != map_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  /// HexagonBuffer copy operations
  void Copy(void *dst, void *src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    HexagonBuffer *hb_src = FindBuffer<HexagonBuffer>(src, bufferMap_);
    HexagonBuffer *hb_dst = FindBuffer<HexagonBuffer>(dst, bufferMap_);

    bool isSrcHb = (hb_src != nullptr);
    bool isDstHb = (hb_dst != nullptr);

    if (isSrcHb && isDstHb) {
      hb_dst->CopyFrom(*hb_src, nbytes);
    } else if (isSrcHb) {
      hb_src->CopyTo(dst, nbytes);
    } else if (isDstHb) {
      hb_dst->CopyFrom(src, nbytes);
    } else {
      CHECK((false), "One of the src/dst should be a hexagon buffer");
    }
  }

private:
  /// Contains the HexagonBuffer objects managed by this class.
  /// Serialises the buffer bookkeeping. A launch can run on several quRT threads
  /// (one program per thread), and the maps below are plain containers, so
  /// concurrent allocate/free corrupts them (measured: dead DSP when VTCM and
  /// multi-threading are both on).
  std::mutex mutex_;

  std::unordered_map<void *, std::unique_ptr<HexagonBuffer>> bufferMap_;

  /// Contains the HexagonBufferAlias objects managed by this class.
  std::unordered_map<void *, std::unique_ptr<hexagon::HexagonBufferAlias>>
      bufferAliasMap_;

  /// Upper bound on the bytes the free cache may hold. Sized to a per-launch
  /// activation set for the HMX paths (S1 is ~1.2 MiB: a 1 MiB accumulator plus
  /// 128 KiB/64 KiB operands) with headroom, while staying a small fraction of a
  /// multi-MiB VTCM pool so live allocations are never starved. This is a perf
  /// heuristic only: allocation failure drops the whole cache and retries, so the
  /// bound cannot turn a would-succeed allocation into a failure.
  static constexpr size_t kMaxCachedBytes = 2u * 1024 * 1024;

  /// Build the cache key from an alloc request. Overloaded on arity: the 3-arg
  /// form is the 1-D request (bytes, alignment, isVtcm), the 4-arg form the 2-D
  /// one (nallocs, bytes, alignment, isVtcm); these are exactly the two
  /// HexagonBuffer constructor shapes.
  static HexagonBuffer::CacheKey MakeCacheKey(size_t nbytes, size_t alignment,
                                              bool isVtcm) {
    return HexagonBuffer::CacheKey{1, nbytes, alignment, isVtcm};
  }
  static HexagonBuffer::CacheKey MakeCacheKey(size_t nallocs, size_t nbytes,
                                              size_t alignment, bool isVtcm) {
    return HexagonBuffer::CacheKey{nallocs, nbytes, alignment, isVtcm};
  }

  /// Freed-but-still-reserved buffers, keyed by footprint. LIFO per key so the
  /// most recently freed block (still the hottest in cache) is handed back first.
  std::unordered_map<HexagonBuffer::CacheKey,
                     std::vector<std::unique_ptr<HexagonBuffer>>,
                     HexagonBufferCacheKeyHash>
      freeCache_;

  /// Bytes currently held by freeCache_, charged on the aligned reservation so it
  /// matches the VTCM the cache is pinning.
  size_t cachedBytes_ = 0;
};

#endif // BUFFERMANAGER_H_
