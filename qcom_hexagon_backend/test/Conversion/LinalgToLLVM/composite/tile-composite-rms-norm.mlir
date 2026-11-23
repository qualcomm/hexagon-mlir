//===- tile-composite-rms-norm.mlir ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

// Tile composite.rms_norm along the batch (non-axis) dimension using scf.for.

// CHECK-LABEL: func.func @tile_for
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK:       scf.for %[[IV:.*]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK:         tensor.extract_slice %{{.*}}[%[[IV]], 0] [32, 1024] [1, 1]
// CHECK:         tensor.extract_slice %{{.*}}[%[[IV]], 0] [32, 1024] [1, 1]
// CHECK:         tensor.extract_slice %{{.*}}[%[[IV]], 0] [32, 1024] [1, 1]
// CHECK:         composite.rms_norm axis = 1
// CHECK-SAME:      tensor<32x1024xf32>
// CHECK:         tensor.insert_slice
func.func @tile_for(%input: tensor<128x1024xf32>,
                     %weights: tensor<128x1024xf32>,
                     %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
  %0 = composite.rms_norm axis = 1 eps = 1.0e-5 norm_size = 1.024000e+03
       %input, %weights, %init
       : tensor<128x1024xf32>, tensor<128x1024xf32>, tensor<128x1024xf32>
       -> tensor<128x1024xf32>
  return %0 : tensor<128x1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["composite.rms_norm"]} in %arg1
         : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [32]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Tile composite.rms_norm into scf.forall.

// CHECK-LABEL: func.func @tile_forall
// CHECK:       scf.forall (%{{.*}}) in (4) shared_outs
// CHECK:         tensor.extract_slice
// CHECK-SAME:      tensor<128x1024xf32> to tensor<32x1024xf32>
// CHECK:         composite.rms_norm axis = 1
// CHECK-SAME:      tensor<32x1024xf32>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
func.func @tile_forall(%input: tensor<128x1024xf32>,
                        %weights: tensor<128x1024xf32>,
                        %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
  %0 = composite.rms_norm axis = 1 eps = 1.0e-5 norm_size = 1.024000e+03
       %input, %weights, %init
       : tensor<128x1024xf32>, tensor<128x1024xf32>, tensor<128x1024xf32>
       -> tensor<128x1024xf32>
  return %0 : tensor<128x1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["composite.rms_norm"]} in %arg1
         : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [32]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
