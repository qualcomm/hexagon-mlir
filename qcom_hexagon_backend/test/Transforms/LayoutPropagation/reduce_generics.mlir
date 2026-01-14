// RUN: linalg-hexagon-opt -split-input-file %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Unpack with i in inner_dims_pos -> linalg.generic with "reduction" in i'th element of iterator_types
//===------------------------------------------------------------------===//

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @reduce_transpose_unpack_with_overlapping_dims(%arg0: tensor<2x2x4x4xi32>) -> tensor<8xi32> {
  %5 = tensor.empty() : tensor<8x8xi32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %5 : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
  %6 = tensor.empty() : tensor<8xi32>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%unpack : tensor<8x8xi32>) outs(%6 : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %8 = arith.maxsi %out, %in : i32
    linalg.yield %8 : i32
  } -> tensor<8xi32>
  return %7 : tensor<8xi32>
}
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: func.func @reduce_transpose_unpack_with_overlapping_dims(%[[A0:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction", "parallel"]
// CHECK-SAME: ins(%[[A0]] : tensor<2x2x4x4xi32>)
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[REDUCE]]
// CHECK-SAME: inner_dims_pos = [0]
// CHECK-SAME: inner_tiles = [4]
// CHECK: return %[[UNPACK]] : tensor<8xi32>

// -----

#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
func.func @reduce_transpose_unpack_with_overlapping_outer_dims(%arg0: tensor<2x2x4x4xi32>) -> tensor<8xi32> {
  %5 = tensor.empty() : tensor<8x8xi32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %5 : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
  %6 = tensor.empty() : tensor<8xi32>
  %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%unpack : tensor<8x8xi32>) outs(%6 : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %8 = arith.maxsi %out, %in : i32
    linalg.yield %8 : i32
  } -> tensor<8xi32>
  return %7 : tensor<8xi32>
}
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: func.func @reduce_transpose_unpack_with_overlapping_outer_dims(%[[A0:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction", "parallel"]
// CHECK-SAME: ins(%[[A0]] : tensor<2x2x4x4xi32>)
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[REDUCE]]
// CHECK-SAME: inner_dims_pos = [0]
// CHECK-SAME: inner_tiles = [4]
// CHECK: return %[[UNPACK]] : tensor<8xi32>