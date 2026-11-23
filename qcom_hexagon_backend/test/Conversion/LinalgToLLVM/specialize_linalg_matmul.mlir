//===- specialize_linalg_matmul.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s --pass-pipeline="builtin.module(hexagon-specialize-linalg-matmul)" | FileCheck %s

#matmul_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#matmul_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#matmul_c = affine_map<(d0, d1, d2) -> (d0, d1)>
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_generic
// CHECK: linalg.matmul
func.func @matmul_generic(
    %lhs: tensor<2x3xf32>,
    %rhs: tensor<3x4xf32>,
    %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %result = linalg.generic {
    indexing_maps = [#matmul_a, #matmul_b, #matmul_c],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>)
    outs(%out : tensor<2x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out_value: f32):
    %product = arith.mulf %lhs_value, %rhs_value : f32
    %sum = arith.addf %out_value, %product : f32
    linalg.yield %sum : f32
  } -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @elementwise_add
// CHECK: linalg.generic
// CHECK-NOT: linalg.add
func.func @elementwise_add(
    %lhs: tensor<2x3xf32>,
    %rhs: tensor<2x3xf32>,
    %out: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = linalg.generic {
    indexing_maps = [#identity_2d, #identity_2d, #identity_2d],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<2x3xf32>, tensor<2x3xf32>)
    outs(%out : tensor<2x3xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out_value: f32):
    %sum = arith.addf %lhs_value, %rhs_value : f32
    linalg.yield %sum : f32
  } -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
