//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_CRUNNERUTILS_H
#define MLIR_EXECUTIONENGINE_CRUNNERUTILS_H

#include <array>
#include <cstdint>

//===----------------------------------------------------------------------===//
// Codegen-compatible structures for StridedMemRef type.
//===----------------------------------------------------------------------===//

/// StridedMemRef descriptor type with static rank.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

//===----------------------------------------------------------------------===//
// Codegen-compatible structure for UnrankedMemRef type.
//===----------------------------------------------------------------------===//
// Unranked MemRef
template <typename T> struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

//===----------------------------------------------------------------------===//
// DynamicMemRefType type.
//===----------------------------------------------------------------------===//
template <typename T> class DynamicMemRefIterator;

// A reference to one of the StridedMemRef types.
template <typename T> class DynamicMemRefType {
public:
  int64_t rank;
  T *basePtr;
  T *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;
  explicit DynamicMemRefType(const ::UnrankedMemRefType<T> &memRef)
      : rank(memRef.rank) {
    auto *desc = static_cast<StridedMemRefType<T, 1> *>(memRef.descriptor);
    basePtr = desc->basePtr;
    data = desc->data;
    offset = desc->offset;
    sizes = rank == 0 ? nullptr : desc->sizes;
    strides = sizes + rank;
  }
};

//===----------------------------------------------------------------------===//
// Small runtime support library for memref.copy lowering during codegen.
//===----------------------------------------------------------------------===//
extern "C" void memrefCopy(int64_t elemSize, ::UnrankedMemRefType<char> *src,
                           ::UnrankedMemRefType<char> *dst);

#endif // MLIR_EXECUTIONENGINE_CRUNNERUTILS_H