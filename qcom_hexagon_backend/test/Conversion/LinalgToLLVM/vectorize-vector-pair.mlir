//===- vectorize-vector-pair.mlir -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-vectorization | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// Elementwise on tensor<64xf32> (vector pair) should be vectorized.

// CHECK-LABEL: func.func @elemwise_64xf32
// CHECK:         vector.transfer_read {{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK:         arith.addf {{.*}} : vector<64xf32>
// CHECK:         vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
func.func @elemwise_64xf32(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %result = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf32>) outs(%b : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// -----

// Reduction from tensor<64xf32> to scalar (vector pair) should be vectorized.

#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @reduction_64xf32
// CHECK:         %[[VEC:.*]] = vector.transfer_read %{{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK:         %[[INIT:.*]] = vector.transfer_read %{{.*}} : tensor<f32>, vector<f32>
// CHECK:         %[[SCALAR:.*]] = vector.extract %[[INIT]][] : f32 from vector<f32>
// CHECK:         vector.multi_reduction <add>, %[[VEC]], %[[SCALAR]] [0] : vector<64xf32> to f32
func.func @reduction_64xf32(%input: tensor<64xf32>,
    %init: tensor<f32>) -> tensor<f32> {
  %result = linalg.generic {
      indexing_maps = [#map2, #map3], iterator_types = ["reduction"]}
      ins(%input : tensor<64xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
