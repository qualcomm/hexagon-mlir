// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s
// This file tests propagation of expand_shape ops across generics with:
// parallel, reduction iterators, transposition affine maps.
// Negative tests check that expand_shape ops are not propagated across
// generics with multiple outputs, when reduced dimension is expanded.
// CHECK: #[[MAP_ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[MAP_REDUCE:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
// CHECK: #[[MAP_LASTDIM:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @single_out
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x21x768xf16>)
// CHECK:         %[[EMPTY0:.+]] = tensor.empty() : tensor<2x21x768xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[EMPTY0]]
// CHECK:         %[[EXP_ARG:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK:         %[[SINGLE_OUT_GEN:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_ID]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXP_ARG]] : tensor<2x7x3x768xf16>) outs(%[[EXP_OUT]] : tensor<2x7x3x768xf32>) {
// CHECK:         ^bb0(%[[IN:.+]]: f16, %[[OUT0:.+]]: f32):
// CHECK:           %[[EXT0:.+]] = arith.extf %[[IN]] : f16 to f32
// CHECK:           linalg.yield %[[EXT0]] : f32
// CHECK:         } -> tensor<2x7x3x768xf32>
// CHECK:         return %[[SINGLE_OUT_GEN]] : tensor<2x7x3x768xf32>

// CHECK-LABEL: func.func @multi_out
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x21x768xf16>)
// CHECK:         %[[EMPTY0:.+]] = tensor.empty() : tensor<2x21x768xf32>
// CHECK:         %[[EMPTY1:.+]] = tensor.empty() : tensor<2x21x768xf64>
// CHECK:         %[[MULTI_OUT_GEN:.+]]:2 = linalg.generic {indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x21x768xf16>) outs(%[[EMPTY0]], %[[EMPTY1]] : tensor<2x21x768xf32>, tensor<2x21x768xf64>) {
// CHECK:         ^bb0(%[[IN:.+]]: f16, %[[OUT0:.+]]: f32, %[[OUT1:.+]]: f64):
// CHECK:           %[[EXT0:.+]] = arith.extf %[[IN]] : f16 to f32
// CHECK:           %[[EXT1:.+]] = arith.extf %[[EXT0]] : f32 to f64
// CHECK:           linalg.yield %[[EXT0]], %[[EXT1]] : f32, f64
// CHECK:         } -> (tensor<2x21x768xf32>, tensor<2x21x768xf64>)
// CHECK:         %[[EXP1:.+]] = tensor.expand_shape %[[MULTI_OUT_GEN]]#0
// CHECK:         %[[EXP2:.+]] = tensor.expand_shape %[[MULTI_OUT_GEN]]#1
// CHECK:         return %[[EXP1]], %[[EXP2]] : tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>

// CHECK-LABEL: func.func @reduction_case
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x21x768xf32>)
// CHECK:         %[[EMPTY0:.+]] = tensor.empty() : tensor<2x21x1xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[EMPTY0]]
// CHECK:         %[[EXP_ARG:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK:         %[[RED_F32:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_REDUCE]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[EXP_ARG]] : tensor<2x7x3x768xf32>) outs(%[[EXP_OUT]] : tensor<2x7x3x1xf32>) {
// CHECK:         ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:           %[[ADD:.+]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:           linalg.yield %[[ADD]] : f32
// CHECK:         } -> tensor<2x7x3x1xf32>
// CHECK:         return %[[RED_F32]] : tensor<2x7x3x1xf32>

// CHECK-LABEL: func.func @elementwise_with_broadcast
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x21x768xf16>, %[[ARG1:.+]]: tensor<768xf16>)
// CHECK:         %[[EMPTY0:.+]] = tensor.empty() : tensor<2x21x768xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[EMPTY0]]
// CHECK:         %[[EXP_ARG:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK:         %[[RES:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_LASTDIM]], #[[MAP_ID]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXP_ARG]], %[[ARG1]] : tensor<2x7x3x768xf16>, tensor<768xf16>) outs(%[[EXP_OUT]] : tensor<2x7x3x768xf32>) {
// CHECK:         ^bb0(%[[IN_F16:.+]]: f16, %[[IN_ARG1:.+]]: f16, %[[OUT:.+]]: f32):
// CHECK:           %[[EXT1:.+]] = arith.extf %[[IN_F16]] : f16 to f32
// CHECK:           %[[EXT2:.+]] = arith.extf %[[IN_ARG1]] : f16 to f32
// CHECK:           %[[MUL:.+]] = arith.mulf %[[EXT1]], %[[EXT2]] : f32
// CHECK:           linalg.yield %[[MUL]] : f32
// CHECK:         } -> tensor<2x7x3x768xf32>
// CHECK:         return %[[RES]] : tensor<2x7x3x768xf32>

// CHECK-LABEL: func.func @transpose_case
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<48x20xf32>) -> tensor<4x5x6x8xf32> {
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<20x48xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[EMPTY]]
// CHECK:         %[[EXP_ARG:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK:         %[[TRANSPOSE:.+]] = linalg.generic {{.*}} ins(%[[EXP_ARG]] : tensor<6x8x4x5xf32>) outs(%[[EXP_OUT]] : tensor<4x5x6x8xf32>)
// CHECK:         return %[[TRANSPOSE]] : tensor<4x5x6x8xf32>

// Negative test case: Expand shape shouldn't be propagated when it would expand a reduction dimension
// CHECK-LABEL: func.func @reduce_first_dim
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<108x3x768xf32>) -> tensor<1x1x3x768xf32> {
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x3x768xf32>
// CHECK:         %[[REDUCED:.+]] = linalg.generic {{.*}} ins(%[[ARG0]] : tensor<108x3x768xf32>) outs(%[[EMPTY]] : tensor<1x3x768xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1], [2], [3]] output_shape [1, 1, 3, 768] : tensor<1x3x768xf32> into tensor<1x1x3x768xf32>
// CHECK:         return %[[EXPANDED]] : tensor<1x1x3x768xf32>

// Positive test case: Expand shape propagated across parallel dimensions only
// CHECK-LABEL: func.func @expand_parallel_only
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<20x30xf32>) -> tensor<4x5x6x5xf32> {
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<20x30xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[EMPTY]] {{\[\[}}0, 1], [2, 3]] output_shape [4, 5, 6, 5] : tensor<20x30xf32> into tensor<4x5x6x5xf32>
// CHECK:         %[[EXP_ARG:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2, 3]] output_shape [4, 5, 6, 5] : tensor<20x30xf32> into tensor<4x5x6x5xf32>
// CHECK:         %[[RESULT:.+]] = linalg.generic {{.*}} ins(%[[EXP_ARG]] : tensor<4x5x6x5xf32>) outs(%[[EXP_OUT]] : tensor<4x5x6x5xf32>)
// CHECK:         return %[[RESULT]] : tensor<4x5x6x5xf32>

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map_transpose_in = affine_map<(d0, d1) -> (d0, d1)>
#map_transpose_out = affine_map<(d0, d1) -> (d1, d0)>
#map_reduce_first_in = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce_first_out = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_parallel = affine_map<(d0, d1) -> (d0, d1)>
module {

  func.func @single_out(%arg0: tensor<2x21x768xf16>) -> tensor<2x7x3x768xf32> {
    %13 = tensor.empty() : tensor<2x21x768xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x21x768xf16>) outs(%13 : tensor<2x21x768xf32>) {
    ^bb0(%in: f16, %out: f32):
      %38 = arith.extf %in : f16 to f32
      linalg.yield %38 : f32
    } -> tensor<2x21x768xf32>
    %exp1 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768]: tensor<2x21x768xf32> into tensor<2x7x3x768xf32>
    return %exp1 : tensor<2x7x3x768xf32>
  }

// Negative test case: Expand shape shouldn't be propagated above multi-out generic
  func.func @multi_out(%arg0: tensor<2x21x768xf16>) -> (tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>) {
    %13 = tensor.empty() : tensor<2x21x768xf32>
    %14 = tensor.empty() : tensor<2x21x768xf64>
    %15:2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x21x768xf16>) outs(%13, %14 : tensor<2x21x768xf32>, tensor<2x21x768xf64>) {
    ^bb0(%in: f16, %out: f32, %out_27: f64):
      %38 = arith.extf %in : f16 to f32
      %39 = arith.extf %38 : f32 to f64
      linalg.yield %38, %39 : f32, f64
    } -> (tensor<2x21x768xf32>, tensor<2x21x768xf64>)
    %exp1 = tensor.expand_shape %15#0 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768]: tensor<2x21x768xf32> into tensor<2x7x3x768xf32>
    %exp2 = tensor.expand_shape %15#1 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768]: tensor<2x21x768xf64> into tensor<2x7x3x768xf64>
    return %exp1, %exp2 : tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>
  }

  func.func @reduction_case(%arg0: tensor<2x21x768xf32>) -> tensor<2x7x3x1xf32> {
    %arg12 = tensor.empty() : tensor<2x21x1xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<2x21x768xf32>) outs(%arg12 : tensor<2x21x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %38 = arith.addf %in, %out : f32
      linalg.yield %38 : f32
    } -> tensor<2x21x1xf32>
    %exp = tensor.expand_shape %18 [[0], [1, 2], [3]] output_shape [2, 7, 3, 1] : tensor<2x21x1xf32> into tensor<2x7x3x1xf32>
    return %exp : tensor<2x7x3x1xf32>
  }

  func.func @elementwise_with_broadcast(%arg0: tensor<2x21x768xf16>, %arg1: tensor<768xf16>) -> tensor<2x7x3x768xf32> {
    %19 = tensor.empty() : tensor<2x21x768xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x21x768xf16>, tensor<768xf16>) outs(%19 : tensor<2x21x768xf32>) {
    ^bb0(%in: f16, %in_29: f16, %out: f32):
      %44 = arith.extf %in : f16 to f32
      %47 = arith.extf %in_29 : f16 to f32
      %48 = arith.mulf %44, %47 : f32
      linalg.yield %48 : f32
    } -> tensor<2x21x768xf32>
    %expanded_8 = tensor.expand_shape %20 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768] : tensor<2x21x768xf32> into tensor<2x7x3x768xf32>
    return %expanded_8: tensor<2x7x3x768xf32>
  }

  func.func @transpose_case(%arg0: tensor<48x20xf32>) -> tensor<4x5x6x8xf32> {
    %empty = tensor.empty() : tensor<20x48xf32>
    %transposed = linalg.generic {indexing_maps = [#map_transpose_in, #map_transpose_out], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<48x20xf32>) outs(%empty : tensor<20x48xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
    } -> tensor<20x48xf32>
    %expanded_out = tensor.expand_shape %transposed [[0, 1], [2, 3]] output_shape [4, 5, 6, 8] : tensor<20x48xf32> into tensor<4x5x6x8xf32>
    return %expanded_out : tensor<4x5x6x8xf32>
  }

  func.func @reduce_first_dim(%arg0: tensor<108x3x768xf32>) -> tensor<1x1x3x768xf32> {
    %empty = tensor.empty() : tensor<1x3x768xf32>
    %reduced = linalg.generic {indexing_maps = [#map_reduce_first_in, #map_reduce_first_out], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : tensor<108x3x768xf32>) outs(%empty : tensor<1x3x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<1x3x768xf32>
    %expanded_out = tensor.expand_shape %reduced [[0, 1], [2], [3]] output_shape [1, 1, 3, 768] : tensor<1x3x768xf32> into tensor<1x1x3x768xf32>
    return %expanded_out : tensor<1x1x3x768xf32>
  }

  func.func @expand_parallel_only(%arg0: tensor<20x30xf32>) -> tensor<4x5x6x5xf32> {
    %empty = tensor.empty() : tensor<20x30xf32>
    %result = linalg.generic {indexing_maps = [#map_parallel, #map_parallel], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<20x30xf32>) outs(%empty : tensor<20x30xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
    } -> tensor<20x30xf32>
    %expanded_out = tensor.expand_shape %result [[0, 1], [2, 3]] output_shape [4, 5, 6, 5] : tensor<20x30xf32> into tensor<4x5x6x5xf32>
    return %expanded_out : tensor<4x5x6x5xf32>
  }

}
