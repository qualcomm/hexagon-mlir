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
#include <unordered_map>
#include <utility>

#include "HexagonBuffer.h"
#include "HexagonCommon.h"

class BufferManager {
public:
  ~BufferManager() {
    if (!bufferMap_.empty()) {
      CHECK((true), "BufferManager is not empty upon destruction");
    }
  }

  /// Free a HexagonBuffer.
  void FreeHexagonBuffer(void *ptr) {
    auto it = bufferMap_.find(ptr);
    CHECK((it != bufferMap_.end()),
          "Attempt made to free unknown or already freed allocation");
    CHECK(it->second != nullptr);
    bufferMap_.erase(it);
  }

  /// Allocate a HexagonBuffer.
  template <typename... Args> void *AllocateHexagonBuffer(Args &&...args) {
    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    void *ptr = buf->GetPointer();
    bufferMap_.insert({ptr, std::move(buf)});
    return ptr;
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
  std::unordered_map<void *, std::unique_ptr<HexagonBuffer>> bufferMap_;
};

#endif // BUFFERMANAGER_H_
