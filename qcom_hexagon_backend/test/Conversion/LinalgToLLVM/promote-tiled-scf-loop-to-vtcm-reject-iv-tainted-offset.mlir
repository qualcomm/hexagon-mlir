//===- promote-tiled-scf-loop-to-vtcm-reject-iv-tainted-offset.mlir -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(promote-tiled-scf-loop-to-vtcm))' | FileCheck %s

// Soundness regression test for `validateRcOffset` / `buildAffineExpr`.
//
// `buildAffineExpr` only handles `arith.muli` when at least one operand is a
// constant; `iv * %dyn` (mul-of-two-vars) falls through to `asSymbol(val)`,
// becoming an opaque AffineSymbolExpr. Symbols cancel in
// `f(iv) - f(iv -> 0)`, so without the iv-taint check the validator would
// conclude the iv-coefficient is exactly the visible `iv * 512` term and
// accept the rc as if its row stride were 512. The actual per-iteration
// offset advance is `64 * (512 + %trick)`, so the rewrite would silently
// mis-index the buffer when `%trick != 0`.
//
// Fix: `dependsOnIV` walks the def chain of any opaque value; if it reaches
// `iv`, `buildAffineExpr` raises an `ivTainted` flag and `validateRcOffset`
// rejects the binding. This test pins that the matcher now bails — the
// scf.for survives unchanged with per-iteration `reinterpret_cast`s and no
// `tiled_generic` tag.

func.func @reject_iv_tainted_offset(%in: memref<*xf32>, %out: memref<*xf32>, %trick: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  scf.for %iv = %c0 to %c1024 step %c64 {
    // Real flat offset = iv * (512 + %trick). The `%iv * %trick` term is
    // mul-of-two-vars — opaque, with iv on its def chain → tainted → reject.
    %t = arith.muli %iv, %trick : index
    %off512 = arith.muli %iv, %c512 : index
    %off = arith.addi %off512, %t : index
    %rc_in = memref.reinterpret_cast %in to
        offset: [%off], sizes: [64, 512], strides: [512, 1]
        : memref<*xf32> to memref<64x512xf32, strided<[512, 1], offset: ?>>
    %tens = bufferization.to_tensor %rc_in restrict
        : memref<64x512xf32, strided<[512, 1], offset: ?>> to tensor<64x512xf32>
    %rc_out = memref.reinterpret_cast %out to
        offset: [%off], sizes: [64, 512], strides: [512, 1]
        : memref<*xf32> to memref<64x512xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %tens in restrict writable %rc_out
        : (tensor<64x512xf32>, memref<64x512xf32, strided<[512, 1], offset: ?>>) -> ()
  }
  return
}

// CHECK-LABEL: func.func @reject_iv_tainted_offset
//
// Rewrite must NOT fire: no full-shape hoisted views, no tiled_generic tag.
// CHECK-NOT:  memref.reinterpret_cast %{{.*}} sizes: [1024, 512]
// CHECK-NOT:  tiled_generic
// CHECK-NOT:  all_parallel
//
// Original per-iteration tile rc is preserved with the iv-tainted offset
// chain still feeding it.
// CHECK:      scf.for %[[IV:[0-9a-zA-Z_]+]] = %{{[0-9a-zA-Z_]+}} to %{{[0-9a-zA-Z_]+}} step %{{[0-9a-zA-Z_]+}} {
// CHECK:        arith.muli %[[IV]], %{{[0-9a-zA-Z_]+}} : index
// CHECK:        arith.muli %[[IV]], %{{[0-9a-zA-Z_]+}} : index
// CHECK:        arith.addi
// CHECK:        memref.reinterpret_cast %{{.*}} sizes: [64, 512]
// CHECK:        bufferization.to_tensor
// CHECK:        memref.reinterpret_cast %{{.*}} sizes: [64, 512]
// CHECK:        bufferization.materialize_in_destination
// CHECK:      }
// CHECK:      return
