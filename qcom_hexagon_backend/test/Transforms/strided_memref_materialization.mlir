//===- strided_memref_materialization.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-materialize-strided-loads -canonicalize | FileCheck %s

// Tests for StridedMemrefMaterializationPattern: a strided 1-D
// memref.reinterpret_cast feeding vector transfers is materialized into a
// contiguous alloc + memref.copy so the transfers use unit-stride loads/stores.

// -----

// Read use: copy-in gather, subview/read rewired to unit stride.
func.func @materialize_strided_for_transfer_read(%base: memref<256xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cast = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
  %sv = memref.subview %cast[0] [32] [1] : memref<64xf32, strided<[4]>> to memref<32xf32, strided<[4]>>
  %v = vector.transfer_read %sv[%c0], %f0 : memref<32xf32, strided<[4]>>, vector<32xf32>
  return %v : vector<32xf32>
}

// CHECK-LABEL: @materialize_strided_for_transfer_read
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
// CHECK:         %[[ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
// CHECK:         memref.copy %[[CAST]], %[[ALLOC]] : memref<64xf32, strided<[4]>> to memref<64xf32>
// CHECK:         %[[SV:.*]] = memref.subview %[[ALLOC]][0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
// CHECK:         %[[V:.*]] = vector.transfer_read %[[SV]][%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<32xf32, strided<[1]>>, vector<32xf32>
// CHECK:         return %[[V]] : vector<32xf32>

// -----

// Negative: unit-stride cast is already contiguous, so the chain is untouched.
func.func @negative_unit_stride(%base: memref<256xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cast = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [1] : memref<256xf32> to memref<64xf32, strided<[1]>>
  %sv = memref.subview %cast[0] [32] [1] : memref<64xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
  %v = vector.transfer_read %sv[%c0], %f0 : memref<32xf32, strided<[1]>>, vector<32xf32>
  return %v : vector<32xf32>
}

// CHECK-LABEL: @negative_unit_stride
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64], strides: [1] : memref<256xf32> to memref<64xf32, strided<[1]>>
// CHECK:         %[[SV:.*]] = memref.subview %[[CAST]][0] [32] [1] : memref<64xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
// CHECK:         vector.transfer_read %[[SV]][%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<32xf32, strided<[1]>>, vector<32xf32>

// -----

// Negative: strided cast feeds only a scalar memref.load, so this
// vectorization-prep pattern does not fire.
func.func @negative_no_vector_use(%base: memref<256xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %cast = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
  %sv = memref.subview %cast[0] [32] [1] : memref<64xf32, strided<[4]>> to memref<32xf32, strided<[4]>>
  %v = memref.load %sv[%c0] : memref<32xf32, strided<[4]>>
  return %v : f32
}

// CHECK-LABEL: @negative_no_vector_use
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
// CHECK:         %[[SV:.*]] = memref.subview %[[CAST]][0] [32] [1] : memref<64xf32, strided<[4]>> to memref<32xf32, strided<[4]>>
// CHECK:         memref.load %[[SV]][%{{.*}}] : memref<32xf32, strided<[4]>>

// -----

// Read-modify-write: copy-in gather before the writer, drain scatter after it,
// both subviews rewired to unit stride.
func.func @materialize_strided_rmw(%base: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %cast = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
  %sv = memref.subview %cast[0] [32] [1] : memref<64xf32, strided<[4]>> to memref<32xf32, strided<[4]>>
  %v = vector.transfer_read %sv[%c0], %f0 : memref<32xf32, strided<[4]>>, vector<32xf32>
  %add = arith.addf %v, %cst : vector<32xf32>
  vector.transfer_write %add, %sv[%c0] : vector<32xf32>, memref<32xf32, strided<[4]>>
  return
}

// CHECK-LABEL: @materialize_strided_rmw
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
// CHECK:         %[[ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
// CHECK:         memref.copy %[[CAST]], %[[ALLOC]] : memref<64xf32, strided<[4]>> to memref<64xf32>
// CHECK:         %[[SV:.*]] = memref.subview %[[ALLOC]][0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
// CHECK:         %[[V:.*]] = vector.transfer_read %[[SV]][%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<32xf32, strided<[1]>>, vector<32xf32>
// CHECK:         %[[ADD:.*]] = arith.addf %[[V]], %{{.*}} : vector<32xf32>
// CHECK:         vector.transfer_write %[[ADD]], %[[SV]][%{{.*}}] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1]>>
// CHECK:         memref.copy %[[ALLOC]], %[[CAST]] : memref<64xf32> to memref<64xf32, strided<[4]>>

// -----

// Pure write: drain scatter after the writer, but NO copy-in (nothing to read,
// and a gather would pull in uninitialized destination memory).
func.func @materialize_strided_pure_write(%base: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %cast = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
  %sv = memref.subview %cast[0] [32] [1] : memref<64xf32, strided<[4]>> to memref<32xf32, strided<[4]>>
  vector.transfer_write %cst, %sv[%c0] : vector<32xf32>, memref<32xf32, strided<[4]>>
  return
}

// CHECK-LABEL: @materialize_strided_pure_write
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64], strides: [4] : memref<256xf32> to memref<64xf32, strided<[4]>>
// CHECK:         %[[ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
// CHECK-NOT:     memref.copy %[[CAST]], %[[ALLOC]]
// CHECK:         %[[SV:.*]] = memref.subview %[[ALLOC]][0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
// CHECK:         vector.transfer_write %{{.*}}, %[[SV]][%{{.*}}] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1]>>
// CHECK:         memref.copy %[[ALLOC]], %[[CAST]] : memref<64xf32> to memref<64xf32, strided<[4]>>
