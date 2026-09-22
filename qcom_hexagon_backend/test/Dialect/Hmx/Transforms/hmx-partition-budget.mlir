//===- hmx-partition-budget.mlir - the pipeline yields to the budget -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The software pipeline needs one crouton-row scratch (one 32 x K f16 tile, the
// pack destination) plus `depth` staging slots (one 32 x K f16 tile each) and a
// status word per slot. The staged path retires the whole activation array, so
// its bytes come back to the room. When even the serial ring (depth 1) plus the
// scratch does not fit the budget the pass is given, the serial tile loop is
// emitted unchanged and a remark names the shortfall: the pipeline is an
// optimisation, never a precondition for correctness. The depth-2 ring that does
// not fit but leaves the depth-1 ring fitting is
// hmx-partition-serial-ring.mlir.
//
// The shape uses Kt=32 so the scratch and each ring slot are a full 32 x 1024
// tile. The committed arrays (activation 2x32x16x32x2 131072, weight
// 2x32x16x32x2 131072, accumulator 2x2x16x32x2 8192, plus the 256-byte
// conversion state) come to 270592 bytes; the staged path frees the 131072-byte
// activation, so the room is `budget - 270592 + 131072 = budget - 139520`. A
// 270592-byte budget leaves 131072 free: the 65536-byte scratch plus the
// 65540-byte serial ring needs 131076 -- four bytes (the status word) more than
// that, so not even the serial ring fits.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=270592}))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=270592}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// Under this budget not even one staging slot fits and the pass says why.
// CHECK-LABEL: func.func @over_budget
// CHECK: hmx.bias_init
// The activation crouton array is still the whole array (the staged path, which
// would retire it, did not apply) ...
// CHECK: memref.alloc() : memref<2x32x16x32x2xf16, 1>
// CHECK: hmx.pack_act ins(%{{.*}} : memref<64x1024xf16>
// ... and the tile loop walks it serially: no staging ring at all.
// CHECK: scf.for
// CHECK: hmx.acc_clear
// CHECK: scf.for
// CHECK: hmx.mma
// CHECK: hmx.acc_read
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: memref.dma_start
// CHECK-NOT: hmx.matmul
func.func @over_budget(%a: memref<64x1024xf16>, %w: memref<1024x64xf16>) {
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
  // expected-remark @+1 {{HMX pipeline not applied: activation staging needs 131076 bytes of VTCM (one crouton scratch plus the serial ring), only 131072 are free}}
  hmx.matmul ins(%ca, %cw : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<2x2x16x32x2xf16, 1>)
  return
}
