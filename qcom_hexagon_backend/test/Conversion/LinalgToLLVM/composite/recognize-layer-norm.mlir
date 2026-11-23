//===- recognize-layer-norm.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-to-composite))' | FileCheck %s

// Test that the layer norm linalg pattern is recognized and replaced with a
// single composite.layer_norm op.  The input IR mirrors the f32 pattern
// produced by Triton -> linalg lowering for the test_layer_norm.py kernel.

// CHECK-LABEL: func.func @layer_norm_f32
// CHECK-SAME:    (%[[INPUT:[a-z0-9]+]]: tensor<1024xf32>, %[[WEIGHTS:[a-z0-9]+]]: tensor<1024xf32>, %[[BIAS:[a-z0-9]+]]: tensor<1024xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %[[LN:.*]] = composite.layer_norm axis = 0 eps = 9.99999974E-6 norm_size = 1.024000e+03 %[[INPUT]], %[[WEIGHTS]], %[[BIAS]], %{{[a-z0-9]+}} : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
// CHECK:         return %[[LN]] : tensor<1024xf32>
// CHECK-NOT:     linalg.generic

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

func.func @layer_norm_f32(%input: tensor<1024xf32>,
                           %weights: tensor<1024xf32>,
                           %bias: tensor<1024xf32>) -> tensor<1024xf32> {
  %cst_zero = arith.constant 0.000000e+00 : f32
  %cst_one = arith.constant 1.000000e+00 : f32
  %cst_N = arith.constant 1.024000e+03 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32

  %empty = tensor.empty() : tensor<1024xf32>

  // mean sum-reduce: sum(x) -> scalar
  %init_scalar = tensor.empty() : tensor<f32>
  %inserted = tensor.insert %cst_zero into %init_scalar[] : tensor<f32>
  %mean_sum = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%input : tensor<1024xf32>)
    outs(%inserted : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<f32>

  // scalar chain: extract -> divf(N) -> mean
  %mean_extracted = tensor.extract %mean_sum[] : tensor<f32>
  %mean = arith.divf %mean_extracted, %cst_N : f32

  // broadcast mean into vector
  %mean_bcast = linalg.fill ins(%mean : f32) outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>

  // sub: x - mean
  %centered = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%input, %mean_bcast : tensor<1024xf32>, tensor<1024xf32>)
    outs(%empty : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %s = arith.subf %in0, %in1 : f32
    linalg.yield %s : f32
  } -> tensor<1024xf32>

  // select: tl.where(mask, centered, zeros) for padding
  %mask = tensor.empty() : tensor<1024xi1>
  %true_val = arith.constant true
  %mask_filled = linalg.fill ins(%true_val : i1) outs(%mask : tensor<1024xi1>) -> tensor<1024xi1>
  %zeros = linalg.fill ins(%cst_zero : f32) outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
  %masked = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%mask_filled, %centered, %zeros : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>)
    outs(%centered : tensor<1024xf32>) {
  ^bb0(%m: i1, %c: f32, %z: f32, %out: f32):
    %sel = arith.select %m, %c, %z : f32
    linalg.yield %sel : f32
  } -> tensor<1024xf32>

  // square: masked_centered * masked_centered
  %sq = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%masked, %masked : tensor<1024xf32>, tensor<1024xf32>)
    outs(%masked : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  // var sum-reduce: sum(centered^2) -> scalar
  %init_scalar2 = tensor.empty() : tensor<f32>
  %inserted2 = tensor.insert %cst_zero into %init_scalar2[] : tensor<f32>
  %var_sum = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%sq : tensor<1024xf32>)
    outs(%inserted2 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<f32>

  // scalar chain: extract -> divf(N) -> addf(eps) -> sqrt -> divf(1,_)
  %var_extracted = tensor.extract %var_sum[] : tensor<f32>
  %variance = arith.divf %var_extracted, %cst_N : f32
  %shifted = arith.addf %variance, %cst_eps : f32
  %sqrt = math.sqrt %shifted : f32
  %rstd = arith.divf %cst_one, %sqrt : f32

  // broadcast rstd into vector
  %rstd_bcast = linalg.fill ins(%rstd : f32) outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>

  // normalize: masked_centered * rstd
  %norm = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%masked, %rstd_bcast : tensor<1024xf32>, tensor<1024xf32>)
    outs(%masked : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  // weight multiply: normalized * weights
  %weighted = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%norm, %weights : tensor<1024xf32>, tensor<1024xf32>)
    outs(%norm : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %m = arith.mulf %in0, %in1 : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>

  // bias add: weighted + bias
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%weighted, %bias : tensor<1024xf32>, tensor<1024xf32>)
    outs(%weighted : tensor<1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %a = arith.addf %in0, %in1 : f32
    linalg.yield %a : f32
  } -> tensor<1024xf32>

  return %result : tensor<1024xf32>
}
