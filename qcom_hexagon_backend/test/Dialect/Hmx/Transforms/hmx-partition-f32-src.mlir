//===- hmx-partition-f32-src.mlir - staging with an f32 row-major source ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The staged tile loop accepts an f32 row-major source, exactly as the f16 one:
// the f32 activation ABI (hmx_pack_act_f32 bulk leaves) made f32 activations
// first-class, but the staging geometry gate and the VTCM slot still assumed an
// f16 source, so every f32-activation matmul -- e.g. llama.cpp's MUL_MAT
// shapes -- silently lost the DMA overlap. The slot is typed as the source's
// own element type and sized accordingly (32 x K x 4 bytes here, twice the f16
// ring), and the pack that reads the slot picks the f32 leaf.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @depth_f32
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// CHECK: %[[SLOT:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf32, 1>
// CHECK: %[[ST:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[T:.*]] = hmx.stage ins(%arg0, %[[ROW:.*]] : memref<128x1024xf32>) outs(%[[SLOT]], %[[ST]] : memref<32x1024xf32, 1>, memref<1xi32>) -> i32
// CHECK: %[[READY:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[SLOT]] : memref<32x1024xf32, 1>) -> memref<32x1024xf32, 1>
// CHECK: hmx.pack_act ins(%[[READY]], {{.*}}, {{.*}} : memref<32x1024xf32, 1>) outs(%[[SCRATCH]] :
// CHECK: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, {{.*}}, {{.*}} {n_croutons = 1 : i32}
// CHECK-NOT: hmx.matmul
func.func @depth_f32(%a: memref<128x1024xf32>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %ca = memref.alloc() : memref<4x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c128 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf32>) outs(%ca : memref<4x32x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>)
  memref.dealloc %ca : memref<4x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}
