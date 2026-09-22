//===- hmx-partition-serial-ring.mlir - the double ring yields to depth 1 -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// When the double-buffered ring does not fit the budget but a single slot does,
// the pass keeps the staged loop and narrows the ring to depth 1: the serial
// source loop (issue -> await -> compute per tile) is left unpipelined, so the
// transfer of tile m+1 overlaps nothing, but the loop is still the
// `hmx.stage`/`hmx.await` form and a remark records the narrowing. The fully
// over-budget case is hmx-partition-budget.mlir.
//
// The committed arrays are 270592 bytes and the staged path frees the
// 131072-byte activation, so the room is `budget - 139520` (see
// hmx-partition-budget.mlir). A 300592-byte budget leaves 161072 free: more than
// the 131076-byte scratch + serial ring, less than the 196616-byte scratch +
// double ring.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=300592}))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=300592}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @serial_ring
// CHECK: hmx.bias_init
// One crouton-row scratch, one slot and one status word: the ring is serial.
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// CHECK: %[[SLOT:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// CHECK: %[[ST:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// CHECK-NOT: memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// The serial staged loop: issue tile m (source row m * 32), await it, compute
// it. No iter_args, no prologue, no epilogue -- the pipeliner never runs at
// depth 1.
// CHECK: scf.for %[[M:.*]] = {{.*}} {
// CHECK: %[[ROW:.*]] = arith.muli %[[M]], {{.*}} : index
// CHECK: %[[T:.*]] = hmx.stage ins(%arg0, %[[ROW]] : memref<64x1024xf16>) outs(%[[SLOT]], %[[ST]] : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// CHECK: %[[READY:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[SLOT]] : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// CHECK: hmx.pack_act ins(%[[READY]], {{.*}}, {{.*}} : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] :
// CHECK: hmx.acc_clear
// CHECK: scf.for %[[K:.*]] = {{.*}} step {{.*}} {
// CHECK: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, {{.*}}, %[[K]] {n_croutons = 1 : i32}
// CHECK: hmx.acc_read {{.*}} {bias_set = 0 : i32}
// The ring and the scratch are released after the loop.
// CHECK: memref.dealloc %[[SLOT]]
// CHECK: memref.dealloc %[[ST]]
// CHECK: memref.dealloc %[[SCRATCH]]
// CHECK-NOT: hmx.matmul
// CHECK-NOT: memref.dma_start
func.func @serial_ring(%a: memref<64x1024xf16>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<64x1024xf16>) outs(%ca : memref<2x32x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  // expected-remark @+1 {{HMX pipeline not applied at depth 2: double-buffered activation staging needs 196616 bytes of VTCM, only 161072 are free; using the serial ring}}
  hmx.matmul ins(%ca, %cw : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<2x2x16x32x2xf16, 1>)
  return
}
