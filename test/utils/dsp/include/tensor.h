//===- tensor.h ------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#define FARF_ALWAYS 1
#include "HAP_compute_res.h"
#include "HAP_farf.h"

#define ENFORCE_SINGLE_PAGE_VTCM_ALLOC 0
#define HAP_REQUEST_TIMEOUT_US 100000

enum MemType { VTCM, HEAP };
enum DataType {
  Half = 1,  // f16
  Float = 2, // f32
  Double = 3,
  Int8 = 4,
  Int16 = 5, // short
  Int32 = 6, // long
  Int64 = 7, // long long
  UInt8 = 8,
  UInt16 = 9,
  UInt32 = 10,
};

template <typename T> int32_t get_dtype() {
  if (std::is_same<T, _Float16>::value) {
    return DataType::Half;
  } else if (std::is_same<T, float>::value) {
    return DataType::Float;
  } else if (std::is_same<T, double>::value) {
    return DataType::Double;
  } else if (std::is_same<T, int8_t>::value) {
    return DataType::Int8;
  } else if (std::is_same<T, int16_t>::value) {
    return DataType::Int16;
  } else if (std::is_same<T, int32_t>::value) {
    return DataType::Int32;
  } else if (std::is_same<T, int64_t>::value) {
    return DataType::Int64;
  } else if (std::is_same<T, uint8_t>::value) {
    return DataType::UInt8;
  } else if (std::is_same<T, uint16_t>::value) {
    return DataType::UInt16;
  } else if (std::is_same<T, uint32_t>::value) {
    return DataType::UInt32;
  } else if (std::is_same<T, int>::value) {
    return DataType::Int32;
  } else {
    throw std::runtime_error("Unsupported tensor's data type");
  }
}

template <int N> struct TensorShape {
  int64_t sizes[N];
  int64_t strides[N];
  int64_t alignment;
};

template <typename T, int N> class Tensor {
  MemRefDescriptor<T, N> memRefDesc;
  MemType mem;
  bool needsAlloc;
  unsigned alignment;
  uint32_t contextID;

  T *alignBuffer(T *buffer, size_t alignment) {
    uintptr_t ptr = (uintptr_t)buffer;
    uintptr_t aligned_ptr = (ptr + alignment - 1) & ~(alignment - 1);
    return (T *)aligned_ptr;
  }

  std::pair<T *, T *> allocateBuffer(int64_t size, MemType memType,
                                     size_t alignment) {
    T *buf = nullptr;
    T *alignedBuf = nullptr;
    switch (memType) {
    case MemType::HEAP:
      buf = (T *)memalign(alignment, size * sizeof(T));
      alignedBuf = buf;
      break;
    case MemType::VTCM:
      compute_res_attr_t compute_res;
      HAP_compute_res_attr_init(&compute_res);

      // Request VTCM
      unsigned int n_bytes = size * sizeof(T);
      HAP_compute_res_attr_set_vtcm_param(&compute_res, n_bytes,
                                          ENFORCE_SINGLE_PAGE_VTCM_ALLOC);
      contextID = 0;
      contextID = HAP_compute_res_acquire(&compute_res, HAP_REQUEST_TIMEOUT_US);
      if (contextID == 0)
        throw std::runtime_error("Failed to acquire VTCM for tensor.");

      buf = (T *)HAP_compute_res_attr_get_vtcm_ptr(&compute_res);
      alignedBuf = alignBuffer(buf, alignment);
      break;
    }
    if (!buf || !alignedBuf)
      throw std::runtime_error("Failed to allocate memory for tensor.");
    return {buf, alignedBuf};
  }

  // Setup descriptor for N > 0
  template <int M = N, std::enable_if_t<M != 0, int> = 0>
  void setupDesc(T *buf, T *alignedBuf, int64_t offset, const int64_t *sizes,
                 const int64_t *strides) {
    memRefDesc.allocated = buf;
    memRefDesc.aligned = alignedBuf;
    memRefDesc.offset = offset;
    memcpy(memRefDesc.sizes, sizes, sizeof(int64_t) * N);
    memcpy(memRefDesc.strides, strides, sizeof(int64_t) * N);
  }

  // Setup descriptor for N == 0
  template <int M = N, std::enable_if_t<M == 0, int> = 0>
  void setupDesc(T *buf, T *alignedBuf, int64_t offset) {
    memRefDesc.allocated = buf;
    memRefDesc.aligned = alignedBuf;
    memRefDesc.offset = offset;
  }

  // This is required for the simulator because of a strange interaction
  // where calling free() in the class destructor requires unnecessary
  // pthread symbols we cannot resolve in the simulator
  void freeBuf() {
    if (!needsAlloc)
      throw std::runtime_error("Cannot free memory which tensor object "
                               "does not own");
    switch (mem) {
    case MemType::HEAP:
      free(memRefDesc.allocated);
      break;
    case MemType::VTCM:
      HAP_compute_res_release(contextID);
      break;
    }
    memRefDesc.allocated = nullptr;
    memRefDesc.aligned = nullptr;
  }

public:
  Tensor(const Tensor &) = delete;

  // Constructor for N > 0
  template <int M = N, std::enable_if_t<M != 0, int> = 0>
  Tensor(const TensorShape<N> &shape, const T *prealloc_buf, MemType mem)
      : mem(mem), needsAlloc(false), alignment(shape.alignment) {
    setupDesc(prealloc_buf, prealloc_buf, 0, shape.sizes, shape.strides);
  }

  // Constructor for N > 0
  template <int M = N, std::enable_if_t<M != 0, int> = 0>
  Tensor(const TensorShape<N> &shape, MemType mem)
      : mem(mem), needsAlloc(true), alignment(shape.alignment) {
    setupDesc(nullptr, nullptr, 0, shape.sizes, shape.strides);
    int64_t size = getElementsCount();
    std::pair<T *, T *> buf_pair = allocateBuffer(size, mem, alignment);
    memRefDesc.allocated = buf_pair.first;
    memRefDesc.aligned = buf_pair.second;
  }

  // Constructor for N > 0
  template <int M = N, std::enable_if_t<M != 0, int> = 0>
  Tensor(MemRefDescriptor<T, N> &desc, MemType mem, int alignment)
      : mem(mem), needsAlloc(false), alignment(alignment) {
    setupDesc(desc.allocated, desc.aligned, desc.offset, desc.sizes,
              desc.strides);
  }

  // Constructor for N == 0
  template <int M = N, std::enable_if_t<M == 0, int> = 0>
  Tensor() : needsAlloc(true), mem(MemType::HEAP), alignment(sizeof(T)) {
    std::pair<T *, T *> buf_pair = allocateBuffer(1, mem, alignment);
    setupDesc(buf_pair.first, buf_pair.second, 0);
  }

  MemRefDescriptor<T, N> *toMemRefDesc() { return &memRefDesc; }

  int64_t getElementsCount() {
    int64_t size = 1;
    if constexpr (N > 0)
      for (int i = 0; i < N; i++)
        size = std::max(size, memRefDesc.sizes[i] * memRefDesc.strides[i]);
    return size;
  }

  int64_t *get_shape() {
    if constexpr (N > 0)
      return memRefDesc.sizes;
    else
      return nullptr;
  }

  int64_t get_rank() { return N; }

  ~Tensor() {
    if (needsAlloc && memRefDesc.allocated)
      freeBuf();
  }

  // Load tensor from file
  void load_from_file(std::string path) {
    FILE *fp = fopen(path.c_str(), "rb");
    int64_t elems = getElementsCount();
    const size_t elems_read = fread(memRefDesc.aligned, sizeof(T), elems, fp);
    if (elems_read != elems) {
      // error handling
      if (feof(fp))
        FARF(ALWAYS, "Error reading file %s: unexpected end of file\n",
             path.c_str());
      else if (ferror(fp))
        FARF(ALWAYS, "[tensor.h] Could not read %d elems from file %s", elems,
             path.c_str());
    }
    fclose(fp);
  }

  // Dump tensor to a file
  // Output file format:
  //   4 byte for tensor datatype +
  //   8 bytes for tensor rank +
  //   if (N > 0) 8 * tensor_rank bytes for tensor shape +
  //   tensor_size * size(tensor_data_type) bytes for tensor data
  void dump_to_file(std::string path) {
    FILE *fp = fopen(path.c_str(), "wb");
    int32_t dtype = get_dtype<T>();
    int64_t elems = getElementsCount();
    int64_t *tensor_shape = memRefDesc.sizes;
    int64_t tensor_rank = N;

    fwrite(&dtype, sizeof(int32_t), 1, fp);
    fwrite(&tensor_rank, sizeof(int64_t), 1, fp);
    if constexpr (N > 0)
      fwrite(tensor_shape, sizeof(int64_t), tensor_rank, fp);
    const size_t elems_written =
        fwrite(memRefDesc.aligned, sizeof(T), elems, fp);
    if (elems_written != elems) {
      if (ferror(fp))
        FARF(ALWAYS, "[tensor.h] Could not write %d elems to file %s", elems,
             path.c_str());
    }
    fclose(fp);
  }
};

#endif // TENSOR_H
