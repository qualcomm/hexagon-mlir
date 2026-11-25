//===- lower-composite-softmax.mlir ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-composite))' | FileCheck %s

// Test that the lowering pass expands memref-form composite.softmax into
// three vectorized scf.for loops with 2x unrolling (step = 64 for f32).

// CHECK-LABEL: func.func @softmax_2d
// CHECK-NOT:     composite.softmax

// Pass 1: max-reduce loop.
// CHECK:         scf.for
// CHECK:           vector.load
// CHECK:           vector.load
// CHECK:           arith.maxnumf
// CHECK:           arith.maxnumf
// CHECK:           scf.yield
// CHECK:         vector.reduction <maxnumf>

// Pass 2: fused exp + sum loop.
// CHECK:         scf.for
// CHECK:           vector.load
// CHECK:           arith.subf
// CHECK:           math.exp
// CHECK:           vector.store
// CHECK:           vector.load
// CHECK:           arith.subf
// CHECK:           math.exp
// CHECK:           vector.store
// CHECK:           arith.addf
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         vector.reduction <add>
// CHECK:         arith.divf

// Pass 3: normalize loop.
// CHECK:         scf.for
// CHECK:           vector.load
// CHECK:           arith.mulf
// CHECK:           vector.store
// CHECK:           vector.load
// CHECK:           arith.mulf
// CHECK:           vector.store

func.func @softmax_2d(%input: memref<4x128xf32>, %output: memref<4x128xf32>) {
  composite.softmax axis = 1 %input, %output : memref<4x128xf32>, memref<4x128xf32>
  return
}
