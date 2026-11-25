//===- double-buffering-s1-dma-check.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-double-buffer-generic-s1))' %s | FileCheck %s

// Test that double-buffering is skipped when a copy can't be lowered to a
// DMA (here, a dynamically shaped tile).
// CHECK-LABEL: func.func @dynamic_tile
// CHECK-NOT: db_generic
func.func @dynamic_tile(%arg0: memref<128x?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %sv = memref.subview %arg0[%i, 0] [64, %n] [1, 1] : memref<128x?xf32> to memref<64x?xf32, strided<[?, 1], offset: ?>>
    %buf = memref.alloc(%n) : memref<64x?xf32, 1>
    memref.copy %sv, %buf : memref<64x?xf32, strided<[?, 1], offset: ?>> to memref<64x?xf32, 1>
    scf.for %j = %c0 to %c64 step %c64 {
    }
    memref.copy %buf, %sv : memref<64x?xf32, 1> to memref<64x?xf32, strided<[?, 1], offset: ?>>
  } {all_parallel, tiled_generic}
  return
}

// Test that a VTCM->VTCM copy after the compute doesn't prevent
// double-buffering; it stays a plain memref.copy.
// CHECK-LABEL: func.func @vtcm_to_vtcm_copy
// CHECK: db_generic
func.func @vtcm_to_vtcm_copy(%arg0: memref<128x32xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %sv = memref.subview %arg0[%i, 0] [64, 32] [1, 1] : memref<128x32xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    %buf = memref.alloc() : memref<64x32xf32, 1>
    memref.copy %sv, %buf : memref<64x32xf32, strided<[32, 1], offset: ?>> to memref<64x32xf32, 1>
    %tmp = memref.alloc() : memref<64x32xf32, 1>
    scf.for %j = %c0 to %c64 step %c64 {
    }
    memref.copy %buf, %tmp : memref<64x32xf32, 1> to memref<64x32xf32, 1>
    memref.copy %tmp, %sv : memref<64x32xf32, 1> to memref<64x32xf32, strided<[32, 1], offset: ?>>
  } {all_parallel, tiled_generic}
  return
}

// Test that double-buffering is skipped when a copy is neither contiguous nor
// 2D strided (here, a non-unit inner stride).
// CHECK-LABEL: func.func @non_unit_inner_stride
// CHECK-NOT: db_generic
func.func @non_unit_inner_stride(%arg0: memref<128x32xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %sv = memref.subview %arg0[%i, 0] [64, 16] [1, 2] : memref<128x32xf32> to memref<64x16xf32, strided<[32, 2], offset: ?>>
    %buf = memref.alloc() : memref<64x16xf32, 1>
    memref.copy %sv, %buf : memref<64x16xf32, strided<[32, 2], offset: ?>> to memref<64x16xf32, 1>
    scf.for %j = %c0 to %c64 step %c64 {
    }
    memref.copy %buf, %sv : memref<64x16xf32, 1> to memref<64x16xf32, strided<[32, 2], offset: ?>>
  } {all_parallel, tiled_generic}
  return
}
