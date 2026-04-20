// RUN: linalg-hexagon-opt %s -div-to-mul-optimization -cse | FileCheck %s

// This test verifies div-to-mul optimization for column-wise broadcast division where
// a 1D tensor is broadcasted along the second dimension (columns) to 2D. This pattern
// complements row-wise broadcasts and is common in normalization operations. The
// optimization computes reciprocals on the 1D tensor before broadcasting.

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @broadcast_div_to_mul_optimization_columnwise
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<256x64xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<64xf32>

// CHECK: %[[RECIP:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<64xf32>)
// CHECK-SAME: outs(%[[EMPTY1]] : tensor<64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[DIV:.*]] = arith.divf %[[CST]], %[[IN]] : f32
// CHECK:   linalg.yield %[[DIV]] : f32

// CHECK: %[[BROADCAST:.*]] = linalg.generic
// CHECK-SAME: ins(%[[RECIP]] : tensor<64xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   linalg.yield %[[IN]] : f32

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg1, %[[BROADCAST]] : tensor<256x64xf32>, tensor<256x64xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN_0]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
// CHECK: return %[[RESULT]]

func.func @broadcast_div_to_mul_optimization_columnwise(%arg0: tensor<64xf32>, %arg1: tensor<256x64xf32>) -> tensor<256x64xf32> {
  %1 = tensor.empty() : tensor<256x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
                      ins(%arg0 : tensor<64xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<256x64xf32>

  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
                       ins(%arg1, %2 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.divf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<256x64xf32>

  return %4 : tensor<256x64xf32>
}
