//===- hexagonmem_lowering_memref_i1_static.mlir --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// Regression: for static memref<1x8xi1>, copy size must be 8 bytes (i1 has
// 8-bit ABI storage on Hexagon), not 1 byte.

// CHECK-LABEL: func.func @copy_i1_static
// CHECK: %[[SZ:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: llvm.call @hexagon_runtime_copy_dsp({{.*}}, %[[SZ]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()

func.func @copy_i1_static(%arg0: memref<1x8xi1>, %arg1: memref<1x8xi1, 1>) {
  hexagonmem.copy %arg0, %arg1 : memref<1x8xi1> to memref<1x8xi1, 1>
  return
}

// -----

// Larger i1 shape: memref<4x8xi1> has 32 elements, each 1 ABI byte -> 32 bytes.
// Confirms numElements * bytesPerElem scales correctly beyond the minimal case.

// CHECK-LABEL: func.func @copy_i1_4x8
// CHECK: %[[SZ:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: llvm.call @hexagon_runtime_copy_dsp({{.*}}, %[[SZ]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()

func.func @copy_i1_4x8(%arg0: memref<4x8xi1>, %arg1: memref<4x8xi1, 1>) {
  hexagonmem.copy %arg0, %arg1 : memref<4x8xi1> to memref<4x8xi1, 1>
  return
}

// -----

// Regression: byte-multiple types must be unaffected by the (bitWidth+7)/8
// ceil formula. f32 (32-bit, 4 bytes/elem) over 64 elements -> 256 bytes.

// CHECK-LABEL: func.func @copy_f32_regression
// CHECK: %[[SZ:.*]] = llvm.mlir.constant(256 : i32) : i32
// CHECK: llvm.call @hexagon_runtime_copy_dsp({{.*}}, %[[SZ]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()

func.func @copy_f32_regression(%arg0: memref<64xf32>, %arg1: memref<64xf32, 1>) {
  hexagonmem.copy %arg0, %arg1 : memref<64xf32> to memref<64xf32, 1>
  return
}

// -----

// Regression: f16 (16-bit, 2 bytes/elem) over 64 elements -> 128 bytes.

// CHECK-LABEL: func.func @copy_f16_regression
// CHECK: %[[SZ:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK: llvm.call @hexagon_runtime_copy_dsp({{.*}}, %[[SZ]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()

func.func @copy_f16_regression(%arg0: memref<64xf16>, %arg1: memref<64xf16, 1>) {
  hexagonmem.copy %arg0, %arg1 : memref<64xf16> to memref<64xf16, 1>
  return
}

// -----

// Dynamic i1 memref: the dynamic path uses getSizeInBytes() which emits
// ptrtoint(getelementptr null+1 of i1) -- LLVM's sizeof(i1). On Hexagon with
// i1:8:8 datalayout this also yields 1 byte/element, consistent with the
// static path. Verify the GEP/ptrtoint sequence is present and the copy size
// is computed as numElems * 1 (not numElems * 0 or numElems/8).

// CHECK-LABEL: func.func @copy_i1_dynamic
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, i1
// CHECK: %[[SIZEOF:.*]] = llvm.ptrtoint %[[GEP]] : !llvm.ptr to i64
// CHECK: %[[TOTAL:.*]] = llvm.mul %{{.*}}, %[[SIZEOF]] : i64
// CHECK: %[[SZ32:.*]] = llvm.trunc %[[TOTAL]] : i64 to i32
// CHECK: llvm.call @hexagon_runtime_copy_dsp({{.*}}, %[[SZ32]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()

func.func @copy_i1_dynamic(%arg0: memref<?xi1>, %arg1: memref<?xi1, 1>) {
  hexagonmem.copy %arg0, %arg1 : memref<?xi1> to memref<?xi1, 1>
  return
}
