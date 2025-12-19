#ifndef BUFFERMANAGER_H_
#define BUFFERMANAGER_H_

#include <cassert>
#include <unordered_map>
#include <utility>

#include "HexagonBuffer.h"
#include "HexagonBufferAlias.h"
#include "HexagonCommon.h"

class HexagonBufferAlias;

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

  /// Returns a pointer to crouton-table that is constructed using `nBytes`
  /// sized `buffer`. The table size is expected to be `nBytes/CROUTON_SIZE`
  void *CreateBufferAlias(void *ptr, size_t nbytes) {
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

  /// Contains the HexagonBufferAlias objects managed by this class.
  std::unordered_map<void *, std::unique_ptr<hexagon::HexagonBufferAlias>>
      bufferAliasMap_;
};

#endif // BUFFERMANAGER_H_
