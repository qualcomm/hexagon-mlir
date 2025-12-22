// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion))' \
// RUN:   | FileCheck %s -check-prefix=FUSION_ON

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @fusion_option_test(%arg0: tensor<4x8x16xf32>,
                                %arg1: tensor<4x8x16xf32>,
				%arg2: tensor<4x8x16xf32>
				) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
        outs(%0 : tensor<4x8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<4x8x16xf32>
    %2 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%1, %arg2 : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
        outs(%0 : tensor<4x8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<4x8x16xf32>
    return %2 : tensor<4x8x16xf32>
  }
}

// FUSION_ON-COUNT-1: linalg.generic
// FUSION_ON-NOT: linalg.generic
