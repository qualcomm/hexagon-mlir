// RUN: linalg-hexagon-opt %s -div-to-mul-optimization -cse | FileCheck %s

// This test verifies div-to-mul optimization when the broadcasted divisor has multiple
// uses. The optimization creates a separate reciprocal broadcast while preserving the
// original broadcast for other uses, ensuring correctness when the same value is used
// in both division and non-division operations.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @broadcast_div_to_mul_opt_many_users
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<256x64xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<256xf32>

// CHECK: %[[RECIP:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<256xf32>)
// CHECK-SAME: outs(%[[EMPTY1]] : tensor<256xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[DIV:.*]] = arith.divf %[[CST]], %[[IN]] : f32
// CHECK:   linalg.yield %[[DIV]] : f32

// CHECK: %[[BROADCAST_RECIP:.*]] = linalg.generic
// CHECK-SAME: ins(%[[RECIP]] : tensor<256xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   linalg.yield %[[IN]] : f32

// CHECK: %[[BROADCAST_ORIG:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<256xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   linalg.yield %[[IN]] : f32

// CHECK: %[[DIV_RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg1, %[[BROADCAST_RECIP]] : tensor<256x64xf32>, tensor<256x64xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN_0]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32

// CHECK: %[[FINAL:.*]] = linalg.generic
// CHECK-SAME: ins(%[[BROADCAST_ORIG]], %[[DIV_RESULT]] : tensor<256x64xf32>, tensor<256x64xf32>)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<256x64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[ADD:.*]] = arith.addf %[[IN]], %[[IN_0]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32
// CHECK: return %[[FINAL]]

func.func @broadcast_div_to_mul_opt_many_users(%X: tensor<256xf32>, %Y: tensor<256x64xf32>) -> tensor<256x64xf32> {
  %1 = tensor.empty() : tensor<256x64xf32>
  %broadcasted = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
                      ins(%X : tensor<256xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<256x64xf32>

  %div = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
                      ins(%Y, %broadcasted : tensor<256x64xf32>, tensor<256x64xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.divf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<256x64xf32>

  %final = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
                      ins(%broadcasted, %div : tensor<256x64xf32>, tensor<256x64xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.addf %in, %in_2 : f32
      linalg.yield %7 : f32
  } -> tensor<256x64xf32>

  return %final : tensor<256x64xf32>
}
