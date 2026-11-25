//===- double-buffering-s2-prologue-match.mlir ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-double-buffer-generic-s2))' %s | FileCheck %s

// Test that the prologue is found by its db_prologue tag, even when another
// scf.if sits between it and the double-buffered loop.
// CHECK-LABEL: func.func @kernel
// CHECK: memref.dma_start
// CHECK: } {db_generic = 0 : i64, db_prologue}
// CHECK-NEXT: scf.if
// CHECK-NOT: memref.dma_start
// CHECK: scf.for
func.func @kernel(%arg0: memref<128x32xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %toggle = memref.alloc() : memref<i1>
  %ping = memref.alloc() : memref<64x32xf32, 1>
  %pong = memref.alloc() : memref<64x32xf32, 1>
  scf.if %cond {
    %sv = memref.subview %arg0[0, 0] [64, 32] [1, 1] : memref<128x32xf32> to memref<64x32xf32, strided<[32, 1]>>
    memref.copy %sv, %ping : memref<64x32xf32, strided<[32, 1]>> to memref<64x32xf32, 1>
  } {db_generic = 0 : i64, db_prologue}
  scf.if %cond {
    memref.store %true, %toggle[] : memref<i1>
  }
  scf.for %i = %c0 to %c128 step %c64 {
    %t = memref.load %toggle[] : memref<i1>
    %next = arith.addi %i, %c64 : index
    %sv = memref.subview %arg0[%i, 0] [64, 32] [1, 1] : memref<128x32xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    %svn = memref.subview %arg0[%next, 0] [64, 32] [1, 1] : memref<128x32xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    scf.if %t {
      scf.if %cond {
        memref.copy %svn, %pong : memref<64x32xf32, strided<[32, 1], offset: ?>> to memref<64x32xf32, 1>
      } {db_prefetch}
      scf.for %j = %c0 to %c64 step %c64 {
      }
      memref.copy %ping, %sv : memref<64x32xf32, 1> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    } {db_ping_kernel}
    %nt = arith.xori %t, %true : i1
    scf.if %nt {
      scf.if %cond {
        memref.copy %svn, %ping : memref<64x32xf32, strided<[32, 1], offset: ?>> to memref<64x32xf32, 1>
      } {db_prefetch}
      scf.for %j = %c0 to %c64 step %c64 {
      }
      memref.copy %pong, %sv : memref<64x32xf32, 1> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    } {db_pong_kernel}
  } {db_generic = 0 : i64}
  return
}
