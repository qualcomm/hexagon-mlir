//===- shard-to-llvm.mlir -------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s --shard-to-llvm | FileCheck %s

// Verify that shard-to-llvm lowers shard ops to arith/tensor using the
// Hexagon ABI: linearIdx = cid * ntpc + tid.
// Function args tail layout: [ntpc, num_cores, reserved, tid, cid, reserved]

module {
  shard.grid @nsp(shape = 4)

  // CHECK-LABEL: func.func @vadd
  // CHECK-NOT: shard.grid
  // CHECK-NOT: shard.sharding
  // CHECK-NOT: shard.all_slice
  // CHECK-NOT: shard.process_linear_index

  // CHECK-DAG: %[[C16:[A-Za-z0-9_]+]] = arith.constant 16 : index
  // CHECK: %[[OFF0:[A-Za-z0-9_]+]] = arith.muli %arg7, %[[C16]] : index
  // CHECK: tensor.extract_slice %arg0[%[[OFF0]]] [16] [1]

  // CHECK: %[[OFF1:[A-Za-z0-9_]+]] = arith.muli %arg7, %[[C16]] : index
  // CHECK: tensor.extract_slice %arg1[%[[OFF1]]] [16] [1]

  // CHECK: linalg.generic

  // CHECK: %[[MUL:[A-Za-z0-9_]+]] = arith.muli %arg7, %arg3 : index
  // CHECK: %[[IDX:[A-Za-z0-9_]+]] = arith.addi %[[MUL]], %arg6 : index
  // CHECK: %[[OFF2:[A-Za-z0-9_]+]] = arith.muli %[[IDX]], %[[C16]] : index
  // CHECK: memref.subview %arg2[%[[OFF2]]] [16] [1]

  func.func @vadd(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %dest: memref<64xf32>,
                  %ntpc: index, %num_cores: index, %reserved0: index,
                  %tid: index, %cid: index, %reserved1: index) {
    %sharding = shard.sharding @nsp split_axes = [[0]] : !shard.sharding
    %all_slice = shard.all_slice %arg0 on @nsp grid_axes = [0] slice_axis = 0
      : tensor<64xf32> -> tensor<16xf32>
    %all_slice_1 = shard.all_slice %arg1 on @nsp grid_axes = [0] slice_axis = 0
      : tensor<64xf32> -> tensor<16xf32>
    %init = tensor.empty() : tensor<16xf32>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%all_slice, %all_slice_1 : tensor<16xf32>, tensor<16xf32>)
      outs(%init : tensor<16xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
    } -> tensor<16xf32>
    %proc = shard.process_linear_index on @nsp : index
    %c16 = arith.constant 16 : index
    %off = arith.muli %proc, %c16 : index
    %sub = memref.subview %dest[%off] [16] [1]
      : memref<64xf32> to memref<16xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %0 in writable %sub
      : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
