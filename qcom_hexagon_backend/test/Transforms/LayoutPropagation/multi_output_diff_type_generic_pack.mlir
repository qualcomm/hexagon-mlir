// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK

func.func @multiresult_generic(%arg0: tensor<10xf16>) -> (tensor<5x2xf32>, tensor<5x2xf64>) {
  %init_0 = tensor.empty() : tensor<10xf32>
  %init_1 = tensor.empty() : tensor<10xf64>
  %genOp:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<10xf16>)
    outs(%init_0, %init_1 : tensor<10xf32>, tensor<10xf64>) {
    ^bb0(%in: f16, %out: f32, %out_1: f64):
      %38 = arith.extf %in : f16 to f32
      %39 = arith.extf %38 : f32 to f64
      linalg.yield %38, %39 : f32, f64
  } -> (tensor<10xf32>, tensor<10xf64>)

  %dest_0 = tensor.empty() : tensor<5x2xf32>
  %packed_0 = linalg.pack %genOp#0
    inner_dims_pos = [0]
    inner_tiles = [2]
    into %dest_0 : tensor<10xf32> -> tensor<5x2xf32>

  %dest_1 = tensor.empty() : tensor<5x2xf64>
  %packed_1 = linalg.pack %genOp#1
    inner_dims_pos = [0]
    inner_tiles = [2]
    into %dest_1 : tensor<10xf64> -> tensor<5x2xf64>

  return %packed_0, %packed_1 : tensor<5x2xf32>, tensor<5x2xf64>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @multiresult_generic(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<10xf16>) -> (tensor<5x2xf32>, tensor<5x2xf64>) {
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<5x2xf64>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<5x2xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.pack %[[ARG0]] inner_dims_pos = [0] inner_tiles = [2] into %[[VAL_2]] : tensor<10xf16> -> tensor<5x2xf16>
// CHECK:           %[[VAL_4:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]] : tensor<5x2xf16>)
// CHECK-SAME:      outs(%[[VAL_0]], %[[VAL_1]] : tensor<5x2xf32>, tensor<5x2xf64>) {
// CHECK-NEXT:      ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f64):
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.extf %[[VAL_5]] : f16 to f32
// CHECK-NEXT:        %[[VAL_9:.*]] = arith.extf %[[VAL_8]] : f32 to f64
// CHECK-NEXT:        linalg.yield %[[VAL_8]], %[[VAL_9]] : f32, f64
// CHECK-NEXT:      } -> (tensor<5x2xf32>, tensor<5x2xf64>)
// CHECK-NEXT:      return %[[VAL_4]]#0, %[[VAL_4]]#1 : tensor<5x2xf32>, tensor<5x2xf64>
// CHECK-NEXT:    }