//===- recognize-rms-norm-2d.mlir -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-to-composite))' | FileCheck %s

// Test that the 2D RMS norm linalg pattern is recognized and replaced with a
// single composite.rms_norm op.  The input IR mirrors the f32 pattern
// produced by Triton -> linalg lowering for the test_block_rms_norm_composite.py
// kernel (block-pointer variant with row-wise partial reduction).
//
// Unlike the 1D pattern (recognize-rms-norm.mlir) where the scalar chain uses
// tensor.extract + scalar arithmetic, the 2D pattern uses 1D linalg.generic
// ops with linalg.fill for constants (divByN, addEps, sqrt, rsqrt).

// CHECK-LABEL: func.func @rms_norm_2d_f32
// CHECK-SAME:    (%[[INPUT:[a-z0-9]+]]: tensor<256x512xf32>, %[[WEIGHTS:[a-z0-9]+]]: tensor<256x512xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         composite.rms_norm axis = 1 eps = 9.99999974E-6 norm_size = 5.120000e+02
// CHECK-SAME:      %[[INPUT]], %[[WEIGHTS]], %{{[a-z0-9]+}}
// CHECK-SAME:      tensor<256x512xf32>, tensor<256x512xf32>, tensor<256x512xf32>
// CHECK-SAME:      -> tensor<256x512xf32>
// CHECK-NOT:     linalg.generic

#map_bcast = affine_map<(d0, d1) -> (d1)>
#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_row = affine_map<(d0, d1) -> (d0)>
#map_1d = affine_map<(d0) -> (d0)>

func.func @rms_norm_2d_f32(%input: tensor<256x512xf32>,
                            %weights: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %cst_one = arith.constant 1.000000e+00 : f32
  %cst_inv_n = arith.constant 0.001953125 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32

  %empty_2d = tensor.empty() : tensor<256x512xf32>
  %empty_1d = tensor.empty() : tensor<256xf32>

  // square: x * x -> 2D
  %sq = linalg.generic {
    indexing_maps = [#map_2d, #map_2d, #map_2d],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %input : tensor<256x512xf32>, tensor<256x512xf32>)
    outs(%empty_2d : tensor<256x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<256x512xf32>

  // sum-reduce: sum(x^2) per row -> 1D
  %fill_zero = linalg.fill ins(%cst_zero : f32) outs(%empty_1d : tensor<256xf32>) -> tensor<256xf32>
  %sum = linalg.generic {
    indexing_maps = [#map_2d, #map_row],
    iterator_types = ["parallel", "reduction"]
  } ins(%sq : tensor<256x512xf32>)
    outs(%fill_zero : tensor<256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<256xf32>

  // divByN: mean_squared = sum / N (via multiply by 1/N)
  %fill_inv_n = linalg.fill ins(%cst_inv_n : f32) outs(%empty_1d : tensor<256xf32>) -> tensor<256xf32>
  %mean_sq = linalg.generic {
    indexing_maps = [#map_1d, #map_1d, #map_1d],
    iterator_types = ["parallel"]
  } ins(%sum, %fill_inv_n : tensor<256xf32>, tensor<256xf32>)
    outs(%empty_1d : tensor<256xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<256xf32>

  // addEps: mean_squared + epsilon
  %fill_eps = linalg.fill ins(%cst_eps : f32) outs(%empty_1d : tensor<256xf32>) -> tensor<256xf32>
  %shifted = linalg.generic {
    indexing_maps = [#map_1d, #map_1d, #map_1d],
    iterator_types = ["parallel"]
  } ins(%mean_sq, %fill_eps : tensor<256xf32>, tensor<256xf32>)
    outs(%empty_1d : tensor<256xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %a = arith.addf %in0, %in1 : f32
    linalg.yield %a : f32
  } -> tensor<256xf32>

  // sqrt
  %sqrt = linalg.generic {
    indexing_maps = [#map_1d, #map_1d],
    iterator_types = ["parallel"]
  } ins(%shifted : tensor<256xf32>)
    outs(%empty_1d : tensor<256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = math.sqrt %in : f32
    linalg.yield %s : f32
  } -> tensor<256xf32>

  // rsqrt: 1.0 / sqrt
  %rsqrt = linalg.generic {
    indexing_maps = [#map_1d, #map_1d],
    iterator_types = ["parallel"]
  } ins(%sqrt : tensor<256xf32>)
    outs(%empty_1d : tensor<256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %r = arith.divf %cst_one, %in : f32
    linalg.yield %r : f32
  } -> tensor<256xf32>

  // broadcast: rsqrt (1D) -> 2D
  %bcast = linalg.generic {
    indexing_maps = [#map_row, #map_2d],
    iterator_types = ["parallel", "parallel"]
  } ins(%rsqrt : tensor<256xf32>)
    outs(%empty_2d : tensor<256x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<256x512xf32>

  // normalize: x * rsqrt_broadcast
  %norm = linalg.generic {
    indexing_maps = [#map_2d, #map_2d, #map_2d],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %bcast : tensor<256x512xf32>, tensor<256x512xf32>)
    outs(%empty_2d : tensor<256x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<256x512xf32>

  // weight mul: normalized * weights
  %result = linalg.generic {
    indexing_maps = [#map_2d, #map_2d, #map_2d],
    iterator_types = ["parallel", "parallel"]
  } ins(%norm, %weights : tensor<256x512xf32>, tensor<256x512xf32>)
    outs(%empty_2d : tensor<256x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<256x512xf32>

  return %result : tensor<256x512xf32>
}
