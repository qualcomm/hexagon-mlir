//===- hexagonmem_weight_resident.mlir - resident weight lowering ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A resident weight alloc does not go through the per-launch allocator: the
// lowering calls the runtime's resident entry with the address of the prepacked
// constant. The address comes from the memref dialect and is lowered to
// `llvm.mlir.addressof` + `llvm.ptrtoint` afterwards.
//
// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @hexagon_runtime_weight_resident_dsp(i64, i32) -> !llvm.ptr

module {
  memref.global "public" constant @__weight : memref<2x2x16x32x2xf16> = dense<1.000000e+00> {alignment = 128 : i64}

  // CHECK-LABEL: func.func @resident
  func.func @resident() {
    // The runtime call replaces the allocator, and its result becomes the
    // buffer's descriptor.
    // CHECK: memref.get_global @__weight : memref<2x2x16x32x2xf16>
    // CHECK: memref.extract_aligned_pointer_as_index
    // CHECK: arith.index_castui
    // CHECK: llvm.call @hexagon_runtime_weight_resident_dsp({{.*}}, {{.*}}) : (i64, i32) -> !llvm.ptr
    // The call's result becomes the buffer's memref descriptor.
    // CHECK: llvm.insertvalue
    %w = hexagonmem.alloc() {alignment = 128 : i64, hmx.weight_resident = {bytes = 8192 : i64, global = @__weight}} : memref<2x2x16x32x2xf16, 1>
    return
  }
}
