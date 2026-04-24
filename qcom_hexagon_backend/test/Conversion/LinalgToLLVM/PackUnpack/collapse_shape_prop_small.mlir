// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s
// This file tests propagation of collapse_shape ops across generics with:
// multiple outputs, reduction iterators (along different dimensions) and
// affine maps with transposition.
// CHECK: #[[MAP_ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[MAP_REDUCE:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
// CHECK: #[[MAP_LASTDIM:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @multi_out
// CHECK:    (%[[ARG0:.+]]: tensor<2x7x3x768xf16>)
// CHECK:         %[[EMPTY0:.+]] = tensor.empty() : tensor<2x21x768xf32>
// CHECK:         %[[EMPTY1:.+]] = tensor.empty() : tensor<2x21x768xf64>
// CHECK:         %[[EMPTY0_EXP:.+]] = tensor.expand_shape %[[EMPTY0]] {{\[\[}}0], [1, 2], [3]] output_shape [2, 7, 3, 768] : tensor<2x21x768xf32> into tensor<2x7x3x768xf32>
// CHECK:         %[[EMPTY1_EXP:.+]] = tensor.expand_shape %[[EMPTY1]] {{\[\[}}0], [1, 2], [3]] output_shape [2, 7, 3, 768] : tensor<2x21x768xf64> into tensor<2x7x3x768xf64>
// CHECK:         %[[MULTI_OUT_GEN:.+]]:2 = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_ID]], #[[MAP_ID]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x7x3x768xf16>) outs(%[[EMPTY0_EXP]], %[[EMPTY1_EXP]] : tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>) {
// CHECK:         ^bb0(%[[IN:.+]]: f16, %[[OUT0:.+]]: f32, %[[OUT1:.+]]: f64):
// CHECK:           %[[EXT0:.+]] = arith.extf %[[IN]] : f16 to f32
// CHECK:           %[[EXT1:.+]] = arith.extf %[[EXT0]] : f32 to f64
// CHECK:           linalg.yield %[[EXT0]], %[[EXT1]] : f32, f64
// CHECK:         } -> (tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>)
// CHECK:         return %[[MULTI_OUT_GEN]]#0, %[[MULTI_OUT_GEN]]#1 : tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>

// CHECK-LABEL: func.func @many_cases
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<108x3x768xf16>, %[[ARG1:.+]]: tensor<12x9x3x768xf32>, %[[ARG2:.+]]: tensor<12x9x3x768xf64>, %[[ARG3:.+]]: f64, %[[ARG4:.+]]: f32, %[[ARG5:.+]]: f64, %[[ARG6:.+]]: tensor<768xf16>) -> tensor<12x9x3x768xf32> {

// CHECK:   %[[EMPTY0:.+]] = tensor.empty() : tensor<108x3x1xf64>
// CHECK:   %[[EMPTY0_EXP:.+]] = tensor.expand_shape %[[EMPTY0]] {{\[\[}}0, 1], [2], [3]] output_shape [12, 9, 3, 1] : tensor<108x3x1xf64> into tensor<12x9x3x1xf64>
// CHECK:   %[[RED_F64:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_REDUCE]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[ARG2]] : tensor<12x9x3x768xf64>) outs(%[[EMPTY0_EXP]] : tensor<12x9x3x1xf64>) {
// CHECK:   ^bb0(%[[IN:.+]]: f64, %[[OUT:.+]]: f64):
// CHECK:     %[[ADD:.+]] = arith.addf %[[IN]], %[[OUT]] : f64
// CHECK:     linalg.yield %[[ADD]] : f64
// CHECK:   } -> tensor<12x9x3x1xf64>

// CHECK:   %[[EMPTY1:.+]] = tensor.empty() : tensor<108x3x1xf64>
// CHECK:   %[[EMPTY1_EXP:.+]] = tensor.expand_shape %[[EMPTY1]] {{\[\[}}0, 1], [2], [3]] output_shape [12, 9, 3, 1] : tensor<108x3x1xf64> into tensor<12x9x3x1xf64>
// CHECK:   %[[MULTI_COLLAPSED:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_REDUCE]], #[[MAP_REDUCE]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[ARG2]], %[[RED_F64]] : tensor<12x9x3x768xf64>, tensor<12x9x3x1xf64>) outs(%[[EMPTY1_EXP]] : tensor<12x9x3x1xf64>) {
// CHECK:   ^bb0(%[[IN1:.+]]: f64, %[[IN2:.+]]: f64, %[[OUT:.+]]: f64):
// CHECK:     %[[DIV:.+]] = arith.divf %[[IN2]], %[[ARG3]] : f64
// CHECK:     %[[SUB:.+]] = arith.subf %[[IN1]], %[[DIV]] : f64
// CHECK:     %[[MUL:.+]] = arith.mulf %[[SUB]], %[[SUB]] : f64
// CHECK:     %[[ADD2:.+]] = arith.addf %[[MUL]], %[[OUT]] : f64
// CHECK:     linalg.yield %[[ADD2]] : f64
// CHECK:   } -> tensor<12x9x3x1xf64>

// CHECK:   %[[EMPTY2:.+]] = tensor.empty() : tensor<108x3x1xf32>
// CHECK:   %[[EMPTY2_EXP:.+]] = tensor.expand_shape %[[EMPTY2]] {{\[\[}}0, 1], [2], [3]] output_shape [12, 9, 3, 1] : tensor<108x3x1xf32> into tensor<12x9x3x1xf32>
// CHECK:   %[[RED_F32:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_REDUCE]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[ARG1]] : tensor<12x9x3x768xf32>) outs(%[[EMPTY2_EXP]] : tensor<12x9x3x1xf32>) {
// CHECK:   ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:     %[[ADD:.+]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:     linalg.yield %[[ADD]] : f32
// CHECK:   } -> tensor<12x9x3x1xf32>

// CHECK:   %[[EMPTY3:.+]] = tensor.empty() : tensor<108x3x768xf32>
// CHECK:   %[[EMPTY3_EXP:.+]] = tensor.expand_shape %[[EMPTY3]] {{\[\[}}0, 1], [2], [3]] output_shape [12, 9, 3, 768] : tensor<108x3x768xf32> into tensor<12x9x3x768xf32>
// CHECK:   %[[EXP_ARG0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2], [3]] output_shape [12, 9, 3, 768] : tensor<108x3x768xf16> into tensor<12x9x3x768xf16>
// CHECK:   %[[RES:.+]] = linalg.generic {indexing_maps = [#[[MAP_ID]], #[[MAP_REDUCE]], #[[MAP_REDUCE]], #[[MAP_LASTDIM]], #[[MAP_ID]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXP_ARG0]], %[[RED_F32]], %[[MULTI_COLLAPSED]], %[[ARG6]] : tensor<12x9x3x768xf16>, tensor<12x9x3x1xf32>, tensor<12x9x3x1xf64>, tensor<768xf16>) outs(%[[EMPTY3_EXP]] : tensor<12x9x3x768xf32>) {
// CHECK:   ^bb0(%[[IN_F16:.+]]: f16, %[[IN_F32:.+]]: f32, %[[IN_F64:.+]]: f64, %[[IN_ARG6:.+]]: f16, %[[OUT:.+]]: f32):
// CHECK:     %[[DIV1:.+]] = arith.divf %[[IN_F64]], %[[ARG3]] : f64
// CHECK:     %[[TRUNC1:.+]] = arith.truncf %[[DIV1]] : f64 to f32
// CHECK:     %[[TRUNC2:.+]] = arith.truncf %[[ARG5]] : f64 to f32
// CHECK:     %[[ADD1:.+]] = arith.addf %[[TRUNC1]], %[[TRUNC2]] : f32
// CHECK:     %[[RSQRT:.+]] = math.rsqrt %[[ADD1]] : f32
// CHECK:     %[[DIV2:.+]] = arith.divf %[[IN_F32]], %[[ARG4]] : f32
// CHECK:     %[[EXT1:.+]] = arith.extf %[[IN_F16]] : f16 to f32
// CHECK:     %[[SUB:.+]] = arith.subf %[[EXT1]], %[[DIV2]] : f32
// CHECK:     %[[MUL1:.+]] = arith.mulf %[[SUB]], %[[RSQRT]] : f32
// CHECK:     %[[EXT2:.+]] = arith.extf %[[IN_ARG6]] : f16 to f32
// CHECK:     %[[MUL2:.+]] = arith.mulf %[[MUL1]], %[[EXT2]] : f32
// CHECK:     linalg.yield %[[MUL2]] : f32
// CHECK:   } -> tensor<12x9x3x768xf32>
// CHECK:   return %[[RES]] : tensor<12x9x3x768xf32>

// CHECK-LABEL: func.func @transpose_case
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<4x5x6x8xf32>) -> tensor<6x8x4x5xf32> {
// CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<48x20xf32>
// CHECK:   %[[EMPTY_EXP:.+]] = tensor.expand_shape %[[EMPTY]] {{\[\[}}0, 1], [2, 3]] output_shape [6, 8, 4, 5] : tensor<48x20xf32> into tensor<6x8x4x5xf32>
// CHECK:   %[[TRANSPOSE:.+]] = linalg.generic {{.*}} ins(%[[ARG0]] : tensor<4x5x6x8xf32>) outs(%[[EMPTY_EXP]] : tensor<6x8x4x5xf32>)
// CHECK:   return %[[TRANSPOSE]] : tensor<6x8x4x5xf32>

// CHECK-LABEL: func.func @reduce_first_dim
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<3x4x5x6xf32>) -> tensor<30xf32> {
// CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x5x6xf32>
// CHECK:   %[[REDUCED:.+]] = linalg.generic {{.*}} ins(%[[ARG0]] : tensor<3x4x5x6xf32>) outs(%[[EMPTY]] : tensor<1x5x6xf32>)
// CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[REDUCED]] {{\[\[}}0, 1, 2]] : tensor<1x5x6xf32> into tensor<30xf32>
// CHECK:   return %[[COLLAPSED]] : tensor<30xf32>

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map_transpose_in = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_transpose_out = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
#map_reduce_first_in = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce_first_out = affine_map<(d0, d1, d2) -> (0, d1, d2)>
module {

  func.func @multi_out(%arg0: tensor<2x7x3x768xf16>) -> (tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>) {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<2x7x3x768xf16> into tensor<2x21x768xf16>
    %13 = tensor.empty() : tensor<2x21x768xf32>
    %14 = tensor.empty() : tensor<2x21x768xf64>
    %15:2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<2x21x768xf16>) outs(%13, %14 : tensor<2x21x768xf32>, tensor<2x21x768xf64>) {
    ^bb0(%in: f16, %out: f32, %out_27: f64):
      %38 = arith.extf %in : f16 to f32
      %39 = arith.extf %38 : f32 to f64
      linalg.yield %38, %39 : f32, f64
    } -> (tensor<2x21x768xf32>, tensor<2x21x768xf64>)
    %exp1 = tensor.expand_shape %15#0 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768]: tensor<2x21x768xf32> into tensor<2x7x3x768xf32>
    %exp2 = tensor.expand_shape %15#1 [[0], [1, 2], [3]] output_shape [2, 7, 3, 768]: tensor<2x21x768xf64> into tensor<2x7x3x768xf64>
    return %exp1, %exp2 : tensor<2x7x3x768xf32>, tensor<2x7x3x768xf64>
  }

  func.func @many_cases(%collapsed: tensor<108x3x768xf16>, %arg0: tensor<12x9x3x768xf32>, %arg1: tensor<12x9x3x768xf64>, %arg13 : f64, %arg14: f32, %arg15: f64, %arg16: tensor<768xf16>) -> (tensor<12x9x3x768xf32>) {
    %collapsed0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<12x9x3x768xf32> into tensor<108x3x768xf32>
    %collapsed1 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<12x9x3x768xf64> into tensor<108x3x768xf64>
    %arg10 = tensor.empty() : tensor<108x3x1xf64>
    %16 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed1 : tensor<108x3x768xf64>) outs(%arg10 : tensor<108x3x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %38 = arith.addf %in, %out : f64
      linalg.yield %38 : f64
    } -> tensor<108x3x1xf64>
    %arg9 = tensor.empty() : tensor<108x3x1xf64>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed1, %16 : tensor<108x3x768xf64>, tensor<108x3x1xf64>) outs(%arg9 : tensor<108x3x1xf64>) {
    ^bb0(%in: f64, %in_27: f64, %out: f64):
      %38 = arith.divf %in_27, %arg13 : f64
      %39 = arith.subf %in, %38 : f64
      %40 = arith.mulf %39, %39 : f64
      %41 = arith.addf %40, %out : f64
      linalg.yield %41 : f64
    } -> tensor<108x3x1xf64>
    %arg12 = tensor.empty() : tensor<108x3x1xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed0 : tensor<108x3x768xf32>) outs(%arg12 : tensor<108x3x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %38 = arith.addf %in, %out : f32
      linalg.yield %38 : f32
    } -> tensor<108x3x1xf32>
    %19 = tensor.empty() : tensor<108x3x768xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map2, #map2, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed, %18, %17, %arg16 : tensor<108x3x768xf16>, tensor<108x3x1xf32>, tensor<108x3x1xf64>, tensor<768xf16>) outs(%19 : tensor<108x3x768xf32>) {
    ^bb0(%in: f16, %in_27: f32, %in_28: f64, %in_29: f16, %out: f32):
      %38 = arith.divf %in_28, %arg13 : f64
      %39 = arith.truncf %38 : f64 to f32
      %40 = arith.truncf %arg15 : f64 to f32
      %41 = arith.addf %39, %40 : f32
      %42 = math.rsqrt %41 : f32
      %43 = arith.divf %in_27, %arg14 : f32
      %44 = arith.extf %in : f16 to f32
      %45 = arith.subf %44, %43 : f32
      %46 = arith.mulf %45, %42 : f32
      %47 = arith.extf %in_29 : f16 to f32
      %48 = arith.mulf %46, %47 : f32
      linalg.yield %48 : f32
    } -> tensor<108x3x768xf32>
    %expanded_8 = tensor.expand_shape %20 [[0, 1], [2], [3]] output_shape [12, 9, 3, 768] : tensor<108x3x768xf32> into tensor<12x9x3x768xf32>
    return %expanded_8: tensor<12x9x3x768xf32>
  }

  func.func @transpose_case(%arg0: tensor<4x5x6x8xf32>) -> tensor<6x8x4x5xf32> {
    %collapsed_in = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<4x5x6x8xf32> into tensor<20x48xf32>
    %empty = tensor.empty() : tensor<48x20xf32>
    %transposed = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%collapsed_in : tensor<20x48xf32>) outs(%empty : tensor<48x20xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
    } -> tensor<48x20xf32>
    %expanded_out = tensor.expand_shape %transposed [[0, 1], [2, 3]] output_shape [6, 8, 4, 5] : tensor<48x20xf32> into tensor<6x8x4x5xf32>
    return %expanded_out : tensor<6x8x4x5xf32>
  }

  func.func @reduce_first_dim(%arg0: tensor<3x4x5x6xf32>) -> tensor<30xf32> {
    %collapsed_in = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<3x4x5x6xf32> into tensor<12x5x6xf32>
    %empty = tensor.empty() : tensor<1x5x6xf32>
    %reduced = linalg.generic {indexing_maps = [#map_reduce_first_in, #map_reduce_first_out], iterator_types = ["reduction", "parallel", "parallel"]} ins(%collapsed_in : tensor<12x5x6xf32>) outs(%empty : tensor<1x5x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<1x5x6xf32>
    %collapsed_out = tensor.collapse_shape %reduced [[0, 1, 2]] : tensor<1x5x6xf32> into tensor<30xf32>
    return %collapsed_out : tensor<30xf32>
  }

}
