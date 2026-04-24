// RUN: linalg-hexagon-opt %s -split-input-file -div-to-mul-optimization -cse -canonicalize | FileCheck %s

// This test verifies div-to-mul optimization for divisions where the divisor is
// broadcasted via linalg.fill operations. The optimization computes the reciprocal
// of the scalar value, fills a tensor with the reciprocal, then performs multiplication
// instead of division. Covers both direct scalar fills and tensor.extract patterns.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @div_to_mul_fill
// CHECK-SAME: (%[[ARG0:.*]]: tensor<128xf32>, %[[FILL_VALUE:.*]]: f32) -> tensor<128xf32>
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128xf32>
// CHECK: %[[RECIP:.*]] = arith.divf %[[CST]], %[[FILL_VALUE]] : f32
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[RECIP]] : f32) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%[[ARG0]], %[[FILLED]] : tensor<128xf32>, tensor<128xf32>)
// CHECK-SAME: outs(%[[ARG0]] : tensor<128xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN_0]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
// CHECK: return %[[RESULT]]

func.func @div_to_mul_fill(%arg0: tensor<128xf32>, %fill_value: f32) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>

  // Fill operation with scalar value
  %filled = linalg.fill ins(%fill_value : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>

  // Generic operation with division
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %filled : tensor<128xf32>, tensor<128xf32>)
    outs(%arg0 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %div = arith.divf %in, %in_4 : f32
    linalg.yield %div : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @div_to_mul_extracted
// CHECK-SAME: (%[[ARG0:.*]]: tensor<128xf32>, %[[ARG1:.*]]: tensor<f32>) -> tensor<128xf32>
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128xf32>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[RECIP:.*]] = arith.divf %[[CST]], %[[EXTRACTED]] : f32
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[RECIP]] : f32) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%[[ARG0]], %[[FILLED]] : tensor<128xf32>, tensor<128xf32>)
// CHECK-SAME: outs(%[[ARG0]] : tensor<128xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN_0]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
// CHECK: return %[[RESULT]]

func.func @div_to_mul_extracted(%arg0: tensor<128xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>

  // Extract scalar from tensor
  %extracted = tensor.extract %arg1[] : tensor<f32>

  // Fill operation with extracted scalar
  %filled = linalg.fill ins(%extracted : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>

  // Generic operation with division
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %filled : tensor<128xf32>, tensor<128xf32>)
    outs(%arg0 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %div = arith.divf %in, %in_4 : f32
    linalg.yield %div : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}
