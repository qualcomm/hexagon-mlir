// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
// This test pushes linalg.unpack to below the multi-result linalg.generic {...extf..}.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @multiuse_generic(%arg0: tensor<1x1x1x24x8x4x32xf16>) -> (tensor<1x8x4x768xf32>, tensor<1x8x4x768xf32>,
  tensor<1x8x4x768xf64>, tensor<1x1x1x24x8x4x32xf32>) {
    %input_unpack_init = tensor.empty() : tensor<1x8x4x768xf16>
    %unpack = linalg.unpack %arg0 inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %input_unpack_init :
    tensor<1x1x1x24x8x4x32xf16> -> tensor<1x8x4x768xf16>
    %output_0_init = tensor.empty() : tensor<1x8x4x768xf32>
    %output_1_init = tensor.empty() : tensor<1x8x4x768xf64>

    %res:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%unpack : tensor<1x8x4x768xf16>) outs(%output_0_init, %output_1_init : tensor<1x8x4x768xf32>, tensor<1x8x4x768xf64>) {
    ^bb0(%in: f16, %out: f32, %out_1: f64):
      %5 = arith.extf %in : f16 to f32
      %6 = arith.extf %5 : f32 to f64
      linalg.yield %5, %6 : f32, f64
    } -> (tensor<1x8x4x768xf32>, tensor<1x8x4x768xf64>)

    %dest_0 = tensor.empty() : tensor<1x1x1x24x8x4x32xf32> 
    %pack = linalg.pack %res#0 inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %dest_0 :
    tensor<1x8x4x768xf32> -> tensor<1x1x1x24x8x4x32xf32>

    return %res#0, %res#0, %res#1, %pack : tensor<1x8x4x768xf32>, tensor<1x8x4x768xf32>, tensor<1x8x4x768xf64>, tensor<1x1x1x24x8x4x32xf32>
}

// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-LABEL:   func.func @multiuse_generic(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x1x24x8x4x32xf16>) ->
// CHECK-SAME:      (tensor<1x8x4x768xf32>, tensor<1x8x4x768xf32>, tensor<1x8x4x768xf64>, tensor<1x1x1x24x8x4x32xf32>) {
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<1x8x4x768xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x8x4x768xf64>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x1x1x24x8x4x32xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x1x1x24x8x4x32xf64>
// CHECK:           %[[VAL_4:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:      ins(%[[ARG0]] : tensor<1x1x1x24x8x4x32xf16>)
// CHECK-SAME:      outs(%[[VAL_2]], %[[VAL_3]] : tensor<1x1x1x24x8x4x32xf32>, tensor<1x1x1x24x8x4x32xf64>) {
// CHECK-NEXT:       ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f64):
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.extf %[[VAL_5]] : f16 to f32
// CHECK-NEXT:        %[[VAL_9:.*]] = arith.extf %[[VAL_8]] : f32 to f64
// CHECK-NEXT:        linalg.yield %[[VAL_8]], %[[VAL_9]] : f32, f64
// CHECK-NEXT:      } -> (tensor<1x1x1x24x8x4x32xf32>, tensor<1x1x1x24x8x4x32xf64>)
// CHECK:           %[[VAL_10:.*]] = linalg.unpack %[[VAL_11:.*]]#0 inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %[[VAL_0]] :
// CHECK-SAME:      tensor<1x1x1x24x8x4x32xf32> -> tensor<1x8x4x768xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.unpack %[[VAL_11]]#1 inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %[[VAL_1]] :
// CHECK-SAME:      tensor<1x1x1x24x8x4x32xf64> -> tensor<1x8x4x768xf64>
// CHECK:           %[[VAL_13:.*]] = tensor.empty() : tensor<1x1x1x24x8x4x32xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.pack %[[VAL_10]] inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %[[VAL_13]] :
// CHECK-SAME:      tensor<1x8x4x768xf32> -> tensor<1x1x1x24x8x4x32xf32>
// CHECK:           return %[[VAL_10]], %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] :
// CHECK-SAME:      tensor<1x8x4x768xf32>, tensor<1x8x4x768xf32>, tensor<1x8x4x768xf64>, tensor<1x1x1x24x8x4x32xf32>
// CHECK:         }
