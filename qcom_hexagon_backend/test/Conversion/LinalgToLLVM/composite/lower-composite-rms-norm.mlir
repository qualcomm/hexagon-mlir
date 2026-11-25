//===- lower-composite-rms-norm.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-composite))' | FileCheck %s

// Test that the lowering pass expands memref-form composite.rms_norm into
// two vectorized scf.for loops with 2x unrolling (step = 64 for f32).

// CHECK-LABEL: func.func @rms_norm_2d
// CHECK-SAME:    (%[[INPUT:.*]]: memref<4x256xf32>, %[[WEIGHTS:.*]]: memref<4x256xf32>, %[[OUTPUT:.*]]: memref<4x256xf32>)
// CHECK-NOT:     composite.rms_norm

// Constants.
// CHECK-DAG:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:     %[[ZERO_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:     %[[NORM_SIZE:.*]] = arith.constant 2.560000e+02 : f32
// CHECK-DAG:     %[[EPS:.*]] = arith.constant 9.99999974E-6 : f32

// Outer loop over rows.
// CHECK:         scf.for

// Loop 1: square + accumulate with 2x unrolling, step = 64.
// CHECK:           scf.for {{.*}} step %[[C64]] iter_args(%{{.*}} = %[[ZERO_VEC]]) -> (vector<32xf32>)
// CHECK:             vector.load %[[INPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             vector.load %[[INPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             arith.addf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             arith.addf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             scf.yield {{.*}} : vector<32xf32>

// Scalar chain: reduction -> mean -> eps -> sqrt -> rsqrt -> broadcast.
// CHECK:           vector.reduction <add>, %{{.*}} fastmath<reassoc> : vector<32xf32> into f32
// CHECK:           arith.divf %{{.*}}, %[[NORM_SIZE]] fastmath<reassoc> : f32
// CHECK:           arith.addf %{{.*}}, %[[EPS]] fastmath<reassoc> : f32
// CHECK:           math.sqrt {{.*}} fastmath<reassoc> : f32
// CHECK:           arith.divf {{.*}} fastmath<reassoc> : f32
// CHECK:           vector.broadcast {{.*}} : f32 to vector<32xf32>

// Loop 2: normalize (x * rsqrt) and weight multiply with 2x unrolling.
// CHECK:           scf.for {{.*}} step %[[C64]]
// CHECK:             vector.load %[[INPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             vector.load %[[WEIGHTS]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             vector.store {{.*}}, %[[OUTPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             vector.load %[[INPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             vector.load %[[WEIGHTS]]{{.*}} : memref<4x256xf32>, vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             arith.mulf {{.*}} fastmath<reassoc> : vector<32xf32>
// CHECK:             vector.store {{.*}}, %[[OUTPUT]]{{.*}} : memref<4x256xf32>, vector<32xf32>

func.func @rms_norm_2d(%input: memref<4x256xf32>,
                        %weights: memref<4x256xf32>,
                        %output: memref<4x256xf32>) {
  composite.rms_norm axis = 1 eps = 1.0e-5 norm_size = 2.560000e+02
      %input, %weights, %output
      : memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>
  return
}
