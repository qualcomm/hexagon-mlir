//===- debug.h ------------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef DEBUG_H
#define DEBUG_H

#include "tensor.h"

template <typename T> void dump_tensor(void *memref) {
  static int tensor_count = 0;

  std::string path = "/vendor/bin/tensor_dump_" +
                     std::to_string(tensor_count++) +
                     ".txt"; // Change the device path if needed
  FILE *fp = fopen(path.c_str(), "wb");

  UnrankedMemrefDesc *tensor = (UnrankedMemrefDesc *)memref;

  int32_t dtype = get_dtype<T>();
  int64_t rank = tensor->rank;
  void *desc_ptr = tensor->ptr;

  T **allocated = (T **)(desc_ptr);
  T **aligned = (T **)((uintptr_t)desc_ptr + sizeof(void *));
  size_t *size =
      (size_t *)((uintptr_t)desc_ptr + 2 * sizeof(void *) + sizeof(int64_t));
  size_t *stride = (size_t *)((uintptr_t)desc_ptr + 2 * sizeof(void *) +
                              sizeof(int64_t) * (1 + rank));

  size_t elems = 1;
  for (size_t i = 0; i < rank; i++) {
    size_t dim_size = *(size + i);
    elems *= dim_size;
  }

  fwrite(&dtype, sizeof(int32_t), 1, fp);
  fwrite(&rank, sizeof(int64_t), 1, fp);
  if (rank > 0)
    fwrite(size, sizeof(size_t), rank, fp);
  size_t elems_written = fwrite(*aligned, sizeof(T), elems, fp);

  if (elems_written != elems) {
    if (ferror(fp))
      FARF(ALWAYS, "[dump.h] Could not write %d elems to file %s", elems,
           path.c_str());
  }

  fclose(fp);
}

extern "C" void _mlir_ciface_dump_tensor_f16(void *memref) {
  dump_tensor<_Float16>(memref);
}
extern "C" void _mlir_ciface_dump_tensor_f32(void *memref) {
  dump_tensor<float>(memref);
}
extern "C" void _mlir_ciface_dump_tensor_i8(void *memref) {
  dump_tensor<int8_t>(memref);
}
extern "C" void _mlir_ciface_dump_tensor_i16(void *memref) {
  dump_tensor<int16_t>(memref);
}
extern "C" void _mlir_ciface_dump_tensor_i32(void *memref) {
  dump_tensor<int32_t>(memref);
}

#endif // DEBUG_H
