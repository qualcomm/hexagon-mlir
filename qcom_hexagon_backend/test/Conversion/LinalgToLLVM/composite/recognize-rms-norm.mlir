//===- recognize-rms-norm.mlir --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-to-composite))' | FileCheck %s

// Test that the RMS norm linalg pattern is recognized and replaced with a
// single composite.rms_norm op.  The input IR mirrors the f32 pattern
// produced by Triton → linalg lowering for the test_rms_norm.py kernel.

// CHECK-LABEL: func.func @rms_norm_f32
// CHECK-SAME:    (%[[INPUT:[a-z0-9]+]]: tensor<1024xf32>, %[[WEIGHTS:[a-z0-9]+]]: tensor<1024xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %[[RMS:.*]] = composite.rms_norm axis = 0 eps = 9.99999974E-6 norm_size = 1.024000e+03 %[[INPUT]], %[[WEIGHTS]], %{{[a-z0-9]+}} : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
// CHECK:         return %[[RMS]] : tensor<1024xf32>
// CHECK-NOT:     linalg.generic

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

func.func @rms_norm_f32(%input: tensor<1024xf32>,
                         %weights: tensor<1024xf32>) -> tensor<1024xf32> {
  %cst_zero = arith.constant 0.000000e+00 : f32
  %cst_one = arith.constant 1.000000e+00 : f32
  %cst_N = arith.constant 1.024000e+03 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32

  %empty = tensor.empty() : tensor<1024xf32>

  // square: x * x
  %sq = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%input, %input : tensor<1024xf32>, tensor<1024xf32>)
    outs(%empty : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  // sum-reduce: sum(x^2) -> scalar
  %init_scalar = tensor.empty() : tensor<f32>
  %inserted = tensor.insert %cst_zero into %init_scalar[] : tensor<f32>
  %sum = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%sq : tensor<1024xf32>)
    outs(%inserted : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<f32>

  // scalar chain: extract -> divf(N) -> addf(eps) -> sqrt
  %extracted = tensor.extract %sum[] : tensor<f32>
  %mean = arith.divf %extracted, %cst_N : f32
  %shifted = arith.addf %mean, %cst_eps : f32
  %sqrt = math.sqrt %shifted : f32

  // rsqrt = 1.0 / sqrt
  %rsqrt = arith.divf %cst_one, %sqrt : f32

  // broadcast: fill rsqrt into vector
  %bcast = linalg.fill ins(%rsqrt : f32) outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>

  // normalize: x * rsqrt
  %empty2 = tensor.empty() : tensor<1024xf32>
  %norm = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%input, %bcast : tensor<1024xf32>, tensor<1024xf32>)
    outs(%empty2 : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  // weight mul: normalized * weights
  %empty3 = tensor.empty() : tensor<1024xf32>
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%norm, %weights : tensor<1024xf32>, tensor<1024xf32>)
    outs(%empty3 : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  return %result : tensor<1024xf32>
}
