//===- outline-linalg-generic.mlir ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt --split-input-file --outline-linalg-generic %s | FileCheck %s

// Each linalg.generic should be outlined into its own private func.func marked
// with the `noinline` passthrough attribute, and replaced by a func.call. The
// outlined functions are emitted before the kernel function.

#map = affine_map<(d0) -> (d0)>

// CHECK-DAG: func.func private @__outlined_generic_0({{.*}}) -> tensor<8xf32> attributes {passthrough = ["noinline"]}
// CHECK-DAG: func.func private @__outlined_generic_1({{.*}}) -> tensor<8xf32> attributes {passthrough = ["noinline"]}

// CHECK-LABEL: func.func @two_generics
// CHECK-SAME:    %[[A:.*]]: tensor<8xf32>
// CHECK:         %[[T0:.*]] = tensor.empty
// CHECK:         %[[C0:.*]] = call @__outlined_generic_0(%[[A]], %[[T0]])
// CHECK:         %[[T1:.*]] = tensor.empty
// CHECK:         %[[C1:.*]] = call @__outlined_generic_1(%[[C0]], %[[T1]])
// CHECK:         return %[[C1]]
// CHECK-NOT:     linalg.generic
func.func @two_generics(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<8xf32>) outs(%e0 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %in : f32
      linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = arith.addf %in, %in : f32
      linalg.yield %a : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// A value captured from above the region (a scalar) must become an extra
// argument of the outlined function and an extra operand of the call.

#map1 = affine_map<(d0) -> (d0)>

// CHECK-DAG: func.func private @__outlined_generic_0({{.*}}f32) -> tensor<4xf32> attributes {passthrough = ["noinline"]}

// CHECK-LABEL: func.func @captured_scalar
// CHECK-SAME:    %[[A:.*]]: tensor<4xf32>, %[[S:.*]]: f32
// CHECK:         %[[E:.*]] = tensor.empty
// CHECK:         call @__outlined_generic_0(%[[A]], %[[E]], %[[S]])
func.func @captured_scalar(%arg0: tensor<4xf32>, %s: f32) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<4xf32>) outs(%e : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %s : f32
      linalg.yield %m : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
