// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(schedule-matmul-for-hvx,linalg-generalize))' | FileCheck %s

func.func @matmul_transpose_b(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>, %arg2: tensor<1024x1024xf32>) {
  %1 = tensor.empty() : tensor<64x1024xf32>
  %transposed = linalg.transpose ins(%arg1 : tensor<1024x64xf32>) outs(%1 : tensor<64x1024xf32>) permutation = [1, 0]
  %2 = linalg.matmul ins(%arg0, %transposed : tensor<1024x64xf32>, tensor<64x1024xf32>) outs(%arg2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return
}

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK: func.func @matmul_transpose_b
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024x64xf32>, %[[ARG1:.*]]: tensor<1024x64xf32>, %[[ARG2:.*]]: tensor<1024x1024xf32>)
// CHECK: %1 = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<1024x64xf32>, tensor<1024x64xf32>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<1024x1024xf32>)

// -----

func.func @matmul_transpose_a(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>, %arg2: tensor<64x64xf32>) {
  %1 = tensor.empty() : tensor<64x1024xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<1024x64xf32>) outs(%1 : tensor<64x1024xf32>) permutation = [1, 0]
  %2 = linalg.matmul ins(%transposed, %arg1 : tensor<64x1024xf32>, tensor<1024x64xf32>) outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return
}

// CHECK: func.func @matmul_transpose_a
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024x64xf32>, %[[ARG1:.*]]: tensor<1024x64xf32>, %[[ARG2:.*]]: tensor<64x64xf32>)
// CHECK: %1 = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP]], #[[MAP1]]], iterator_types = ["reduction", "parallel", "parallel"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<1024x64xf32>, tensor<1024x64xf32>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<64x64xf32>)

// -----

func.func @batch_matmul_transpose_b(%arg0: tensor<12x1024x64xf32>, %arg1: tensor<12x1024x64xf32>, %arg2: tensor<12x1024x1024xf32>) {
  %1 = tensor.empty() : tensor<12x64x1024xf32>
  %transposed = linalg.transpose ins(%arg1 : tensor<12x1024x64xf32>) outs(%1 : tensor<12x64x1024xf32>) permutation = [0, 2, 1]
  %2 = linalg.batch_matmul ins(%arg0, %transposed : tensor<12x1024x64xf32>, tensor<12x64x1024xf32>) outs(%arg2 : tensor<12x1024x1024xf32>) -> tensor<12x1024x1024xf32>
  return
}

// CHECK: func.func @batch_matmul_transpose_b
// CHECK-SAME: (%[[ARG0:.*]]: tensor<12x1024x64xf32>, %[[ARG1:.*]]: tensor<12x1024x64xf32>, %[[ARG2:.*]]: tensor<12x1024x1024xf32>)
// CHECK: %1 = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<12x1024x64xf32>, tensor<12x1024x64xf32>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<12x1024x1024xf32>)

// -----

func.func @batch_matmul_transpose_a(%arg0: tensor<12x1024x64xf32>, %arg1: tensor<12x1024x64xf32>, %arg2: tensor<12x64x64xf32>) {
  %1 = tensor.empty() : tensor<12x64x1024xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<12x1024x64xf32>) outs(%1 : tensor<12x64x1024xf32>) permutation = [0, 2, 1]
  %2 = linalg.batch_matmul ins(%transposed, %arg1 : tensor<12x64x1024xf32>, tensor<12x1024x64xf32>) outs(%arg2 : tensor<12x64x64xf32>) -> tensor<12x64x64xf32>
  return
}

// CHECK: func.func @batch_matmul_transpose_a
// CHECK-SAME: (%[[ARG0:.*]]: tensor<12x1024x64xf32>, %[[ARG1:.*]]: tensor<12x1024x64xf32>, %[[ARG2:.*]]: tensor<12x64x64xf32>)
// CHECK: %1 = linalg.generic {indexing_maps = [#[[MAP5]], #[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<12x1024x64xf32>, tensor<12x1024x64xf32>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<12x64x64xf32>)
