//===- tile-composite-softmax.mlir ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

// Tile composite.softmax along the batch (non-axis) dimension using scf.for.

// CHECK-LABEL: func.func @tile_for
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]]
// CHECK:         tensor.extract_slice %{{.*}}[%[[IV]], 0] [64, 512] [1, 1]
// CHECK:         tensor.extract_slice %{{.*}}[%[[IV]], 0] [64, 512] [1, 1]
// CHECK:         composite.softmax axis = 1
// CHECK-SAME:      tensor<64x512xf32>
// CHECK:         tensor.insert_slice
func.func @tile_for(%arg0: tensor<256x512xf32>,
                     %arg1: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %0 = composite.softmax axis = 1 %arg0, %arg1
       : tensor<256x512xf32>, tensor<256x512xf32> -> tensor<256x512xf32>
  return %0 : tensor<256x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["composite.softmax"]} in %arg1
         : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [64]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Tile composite.softmax into scf.forall (the path used by FormVirtualThreads).

// CHECK-LABEL: func.func @tile_forall
// CHECK:       scf.forall (%{{.*}}) in (4) shared_outs
// CHECK:         tensor.extract_slice
// CHECK-SAME:      tensor<256x512xf32> to tensor<64x512xf32>
// CHECK:         composite.softmax axis = 1
// CHECK-SAME:      tensor<64x512xf32>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
func.func @tile_forall(%arg0: tensor<256x512xf32>,
                        %arg1: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %0 = composite.softmax axis = 1 %arg0, %arg1
       : tensor<256x512xf32>, tensor<256x512xf32> -> tensor<256x512xf32>
  return %0 : tensor<256x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["composite.softmax"]} in %arg1
         : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [64]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Non-divisible tile size: the framework should emit affine.min clamping.

// CHECK-LABEL: func.func @tile_nondivisible
// CHECK:       scf.for
// CHECK:         affine.min
// CHECK:         tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%{{.*}}, 512]
// CHECK:         composite.softmax axis = 1
// CHECK-SAME:      tensor<?x512xf32>
func.func @tile_nondivisible(%arg0: tensor<100x512xf32>,
                              %arg1: tensor<100x512xf32>) -> tensor<100x512xf32> {
  %0 = composite.softmax axis = 1 %arg0, %arg1
       : tensor<100x512xf32>, tensor<100x512xf32> -> tensor<100x512xf32>
  return %0 : tensor<100x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["composite.softmax"]} in %arg1
         : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [64]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
