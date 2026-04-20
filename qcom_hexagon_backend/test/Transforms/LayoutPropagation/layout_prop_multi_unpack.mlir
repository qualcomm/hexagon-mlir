// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=true}))' | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//===------------------------------------------------------------------===//
// 1) Positive: multiple unpacked inputs into generic, 
//    same tiling + same indexing maps, pattern should be matched.
//===------------------------------------------------------------------===//
func.func @multi_unpack_inputs_ok(
    %arg0: tensor<12x2x56x56x32xf32>,
    %arg1: tensor<12x2x56x56x32xf32>
) -> tensor<12x56x56x64xf32> {
    // Two unpacks with same config.
    %empty0 = tensor.empty() : tensor<12x56x56x64xf32>
    %empty1 = tensor.empty() : tensor<12x56x56x64xf32>
    %empty2 = tensor.empty() : tensor<12x56x56x64xf32>
    %unpack0 = linalg.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %empty0
        : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
    %unpack1 = linalg.unpack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %empty1
        : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins (%unpack0, %unpack1: tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>) outs(%empty2 : tensor<12x56x56x64xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
            %sum = arith.addf %in0, %in1 : f32
            linalg.yield %sum : f32
    } -> tensor<12x56x56x64xf32>
    return %result : tensor<12x56x56x64xf32>
}

//  CHECK-LABEL: func.func @multi_unpack_inputs_ok(
//  CHECK: %[[EMPTY0:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
//  CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32
//  CHECK: %[[GENERIC_OUT:.+]] = linalg.generic
//  CHECK-SAME: ins(%[[ARG0:.+]], %[[ARG1:.+]] : tensor<12x2x56x56x32xf32>, tensor<12x2x56x56x32xf32>) outs(%[[EMPTY1]] : tensor<12x2x56x56x32xf32>) {
//  CHECK: %[[UNPACK0:.+]] = linalg.unpack %[[GENERIC_OUT]]
//  CHECK-SAME: outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[EMPTY0]] : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
//  CHECK: return %[[UNPACK0]] : tensor<12x56x56x64xf32>

// -----
//===------------------------------------------------------------------===//
// 2) Negative: multiple unpacked inputs into generic, DIFFERENT tiling, pattern should NOT be matched.
//===------------------------------------------------------------------===//
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @multi_unpack_inputs_diff_tiling(
    %arg0: tensor<12x2x56x56x32xf32>,
    %arg1: tensor<12x4x56x56x16xf32>
) -> tensor<12x56x56x64xf32> {
    // Two unpacks with different tiling.
    %empty0 = tensor.empty() : tensor<12x56x56x64xf32>
    %empty1 = tensor.empty() : tensor<12x56x56x64xf32>
    %empty2 = tensor.empty() : tensor<12x56x56x64xf32>
    %unpack0 = linalg.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %empty0
        : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
    %unpack1 = linalg.unpack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [16] into %empty1
        : tensor<12x4x56x56x16xf32> -> tensor<12x56x56x64xf32>
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins (%unpack0, %unpack1: tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>) outs(%empty2 : tensor<12x56x56x64xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
            %sum = arith.addf %in0, %in1 : f32
            linalg.yield %sum : f32
    } -> tensor<12x56x56x64xf32>
    return %result : tensor<12x56x56x64xf32>
}

//  CHECK-LABEL: func.func @multi_unpack_inputs_diff_tiling(
//  CHECK: %[[EMPTY0:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
//  CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
//  CHECK: %[[EMPTY2:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
//  CHECK: %[[UNPACK0:.+]] = linalg.unpack %[[ARG0]]
//  CHECK: %[[UNPACK1:.+]] = linalg.unpack %[[ARG1]]
//  CHECK: %[[GENERIC_OUT:.+]] = linalg.generic
//  CHECK-SAME: ins(%[[UNPACK0]], %[[UNPACK1]] : tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>) outs(%[[EMPTY2]] : tensor<12x56x56x64xf32>) {
//  CHECK: return %[[GENERIC_OUT]] : tensor<12x56x56x64xf32>

// -----
//===------------------------------------------------------------------===//
// 3) Negative: multiple unpacked inputs into generic, DIFFERENT indexing maps, pattern should NOT be matched.
//===------------------------------------------------------------------===//
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#diff_map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
func.func @multi_unpack_inputs_diff_indexing(
    %arg0: tensor<2x2x2x2x2x2x2x2xf32>,
    %arg1: tensor<2x2x2x2x2x2x2x2xf32>
) -> tensor<4x4x4x4xf32> {
    // Two unpacks with different indexing maps.
    %empty0 = tensor.empty() : tensor<4x4x4x4xf32>
    %empty1 = tensor.empty() : tensor<4x4x4x4xf32>
    %empty2 = tensor.empty() : tensor<4x4x4x4xf32>
    %unpack0 = linalg.unpack %arg0 
    inner_dims_pos = [0, 1, 2, 3] inner_tiles = [2, 2, 2, 2] into %empty0
        : tensor<2x2x2x2x2x2x2x2xf32> -> tensor<4x4x4x4xf32>
    %unpack1 = linalg.unpack %arg1 
    inner_dims_pos = [0, 1, 2, 3] inner_tiles = [2, 2, 2, 2] into %empty1
        : tensor<2x2x2x2x2x2x2x2xf32> -> tensor<4x4x4x4xf32>
    %result = linalg.generic {indexing_maps = [#map, #diff_map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    ins (%unpack0, %unpack1: tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) outs(%empty2 : tensor<4x4x4x4xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
            %sum = arith.addf %in0, %in1 : f32
            linalg.yield %sum : f32
    } -> tensor<4x4x4x4xf32>
    return %result : tensor<4x4x4x4xf32>
}
//  CHECK-LABEL: func.func @multi_unpack_inputs_diff_indexing(
//  CHECK: %[[EMPTY0:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//  CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//  CHECK: %[[EMPTY2:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//  CHECK: %[[UNPACK0:.+]] = linalg.unpack %[[ARG0]]
//  CHECK: %[[UNPACK1:.+]] = linalg.unpack %[[ARG1]]
//  CHECK: %[[GENERIC_OUT:.+]] = linalg.generic
//  CHECK-SAME: ins(%[[UNPACK0]], %[[UNPACK1]] : tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) outs(%[[EMPTY2]] : tensor<4x4x4x4xf32>) {
//  CHECK: return %[[GENERIC_OUT]] : tensor<4x4x4x4xf32>
