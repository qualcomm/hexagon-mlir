// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d1, d0)> 
module attributes {llvm.target_triple = "hexagon"} {

  // CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
  // CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
  // CHECK-LABEL: @fp16_transpose
  // CHECK-SAME: (%arg0: tensor<1024x64xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024xf32>
  // CHECK: %0 = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} 
  // CHECK-SAME: ins(%arg0 : tensor<1024x64xf32>) outs(%arg1 : tensor<64x1024xf32>)

  func.func @fp16_transpose(%arg0: tensor<1024x64xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %transposed = linalg.transpose ins(%arg0 : tensor<1024x64xf32>) outs(%arg1 : tensor<64x1024xf32>) permutation = [1, 0]
    return %transposed : tensor<64x1024xf32>
  }
}
