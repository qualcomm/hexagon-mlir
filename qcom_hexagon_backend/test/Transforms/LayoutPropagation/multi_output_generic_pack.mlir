// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
// This test moves linalg.pack to above the multi-restult linalg.generic.

func.func @multiresult_generic(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<5x2xf32>, tensor<5x2xf32>) {
  %sum_init = tensor.empty() : tensor<10xf32>
  %prod_init = tensor.empty() : tensor<10xf32>

  %sum_and_prod:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, // for %arg0
      affine_map<(d0) -> (d0)>, // for %arg1
      affine_map<(d0) -> (d0)>, // for sum_init
      affine_map<(d0) -> (d0)>  // for prod_init
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
    outs(%sum_init, %prod_init : tensor<10xf32>, tensor<10xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %sum_acc: f32, %prod_acc: f32):
    %sum = arith.addf %lhs, %rhs : f32
    %prod = arith.mulf %lhs, %rhs : f32
    linalg.yield %sum, %prod : f32, f32
  } -> (tensor<10xf32>, tensor<10xf32>)

  %dest_0 = tensor.empty() : tensor<5x2xf32>
  %packed_0 = linalg.pack %sum_and_prod#0
    inner_dims_pos = [0]
    inner_tiles = [2]
    into %dest_0 : tensor<10xf32> -> tensor<5x2xf32>

  %dest_1 = tensor.empty() : tensor<5x2xf32>
  %packed_1 = linalg.pack %sum_and_prod#1
    inner_dims_pos = [0]
    inner_tiles = [2]
    into %dest_1 : tensor<10xf32> -> tensor<5x2xf32>

  return %packed_0, %packed_1 : tensor<5x2xf32>, tensor<5x2xf32>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @multiresult_generic(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<10xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<10xf32>) -> (tensor<5x2xf32>, tensor<5x2xf32>) {
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.pack %[[ARG0]] inner_dims_pos = [0] inner_tiles = [2] into %[[VAL_2]] : tensor<10xf32> -> tensor<5x2xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<5x2xf32>
// CHECK:           %[[VAL_5:.*]] = linalg.pack %[[ARG1]] inner_dims_pos = [0] inner_tiles = [2] into %[[VAL_4]] : tensor<10xf32> -> tensor<5x2xf32>
// CHECK:           %[[VAL_6:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]], %[[VAL_5]] : tensor<5x2xf32>, tensor<5x2xf32>)
// CHECK-SAME:      outs(%[[VAL_0]], %[[VAL_1]] : tensor<5x2xf32>, tensor<5x2xf32>) {
// CHECK-NEXT:      ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32, %[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: f32):
// CHECK-NEXT:        %[[VAL_11:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK-NEXT:        %[[VAL_12:.*]] = arith.mulf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK-NEXT:        linalg.yield %[[VAL_11]], %[[VAL_12]] : f32, f32
// CHECK-NEXT:      } -> (tensor<5x2xf32>, tensor<5x2xf32>)
// CHECK-NEXT:      return %[[VAL_6]]#0, %[[VAL_6]]#1 : tensor<5x2xf32>, tensor<5x2xf32>
// CHECK-NEXT:    }