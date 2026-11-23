//===- memref_runtime.h --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_MEMREF_RUNTIME_H
#define HEXAGON_MEMREF_RUNTIME_H

#include <stdint.h>

template <typename T, int N> struct MemRefDescriptor {
  union {
    T *allocated;
    T *basePtr;
  };
  union {
    T *aligned;
    T *data;
  };
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename T> struct MemRefDescriptor<T, 0> {
  union {
    T *allocated;
    T *basePtr;
  };
  union {
    T *aligned;
    T *data;
  };
  int64_t offset;
};

template <typename T, int N> using StridedMemRefType = MemRefDescriptor<T, N>;

struct UnrankedMemRefDescriptor {
  int64_t rank;
  union {
    void *descriptor;
    void *ptr;
  };
};

template <typename T> using UnrankedMemRefType = UnrankedMemRefDescriptor;

using UnrankedMemrefDesc = UnrankedMemRefDescriptor;

template <typename T> class DynamicMemRefView {
public:
  int64_t rank;
  T *basePtr;
  T *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;

  explicit DynamicMemRefView(const StridedMemRefType<T, 0> &memRef)
      : rank(0), basePtr(memRef.basePtr), data(memRef.data),
        offset(memRef.offset), sizes(nullptr), strides(nullptr) {}

  template <int N>
  explicit DynamicMemRefView(const StridedMemRefType<T, N> &memRef)
      : rank(N), basePtr(memRef.basePtr), data(memRef.data),
        offset(memRef.offset), sizes(memRef.sizes), strides(memRef.strides) {}

  explicit DynamicMemRefView(const UnrankedMemRefType<T> &memRef)
      : rank(memRef.rank) {
    auto *desc = static_cast<StridedMemRefType<T, 1> *>(memRef.descriptor);
    basePtr = desc->basePtr;
    data = desc->data;
    offset = desc->offset;
    sizes = rank == 0 ? nullptr : desc->sizes;
    strides = rank == 0 ? nullptr : desc->sizes + rank;
  }
};

typedef MemRefDescriptor<float, 4> Ty_MemRefDescFloat4D;
typedef MemRefDescriptor<float, 3> Ty_MemRefDescFloat3D;
typedef MemRefDescriptor<float, 2> Ty_MemRefDescFloat2D;
typedef MemRefDescriptor<float, 1> Ty_MemRefDescFloat1D;
typedef MemRefDescriptor<_Float16, 1> Ty_MemRefDescHalf1D;
typedef MemRefDescriptor<_Float16, 2> Ty_MemRefDescHalf2D;
typedef MemRefDescriptor<_Float16, 3> Ty_MemRefDescHalf3D;
typedef MemRefDescriptor<_Float16, 4> Ty_MemRefDescHalf4D;

#endif // HEXAGON_MEMREF_RUNTIME_H
