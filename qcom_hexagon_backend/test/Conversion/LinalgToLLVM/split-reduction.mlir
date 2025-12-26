// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hexagon-tiling{enable-split-reduction=true})' | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
func.func @split_reduction(%input: tensor<256x64xf32>, %output: tensor<256xf32>) -> tensor<256xf32> {
 %reduced = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%input : tensor<256x64xf32>) outs(%output : tensor<256xf32>) {
 ^bb0(%in: f32, %out: f32):
 %result = arith.addf %in, %out : f32
 linalg.yield %result : f32
 } -> tensor<256xf32>
 return %reduced : tensor<256xf32>
}
// CHECK-LABEL: func.func @split_reduction(
// CHECK-SAME: %[[ARG0:.*]]: tensor<256x64xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<256xf32>) -> tensor<256xf32> {
// CHECK: %[[VAL_3:.*]] = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %{{.*}}) -> (tensor<256x32xf32>) {
// CHECK: %[[VAL_4:.*]] = tensor.extract_slice %[[ARG0]][0, %[[ARG2]]] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
// CHECK: %[[VAL_5:.*]] = tensor.extract_slice %[[ARG3]][0, 0] [256, 32] [1, 1] : tensor<256x32xf32> to tensor<256x32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_4]], %[[VAL_5]] : tensor<256x32xf32>, tensor<256x32xf32>) outs(%[[VAL_5]] : tensor<256x32xf32>) {
// CHECK: ^bb0(%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK: %[[VAL_14:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f32
// CHECK: linalg.yield %[[VAL_14]] : f32
// CHECK: } -> tensor<256x32xf32>
// CHECK: %[[VAL_7:.*]] = tensor.insert_slice %[[VAL_6]] into %[[ARG3]][0, 0] [256, 32] [1, 1] : tensor<256x32xf32> into tensor<256x32xf32>
// CHECK: scf.yield %[[VAL_7]] : tensor<256x32xf32>
// CHECK: }
// CHECK: %reduced = linalg.reduce ins(%[[VAL_3]] : tensor<256x32xf32>) outs(%[[ARG1]] : tensor<256xf32>) dimensions = [1]
// CHECK: (%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32) {
// CHECK: %[[VAL_14:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f32
// CHECK: linalg.yield %[[VAL_14]] : f32
// CHECK: return %reduced : tensor<256xf32>
