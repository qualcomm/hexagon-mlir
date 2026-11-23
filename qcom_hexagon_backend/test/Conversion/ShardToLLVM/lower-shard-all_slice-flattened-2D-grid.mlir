//===- lower-shard-all_slice-flattened-2D-grid.mlir -----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// Test: shard.all_slice with grid_axes = [0, 1] lowers to
// tensor.extract_slice using the flattened participant index.
//
// For the Hexagon ABI tail:
//
//   [..., ntpc, num_cores, reserved0, tid, cid, reserved1]
//
// this means:
//
//   linearIdx = cid * ntpc + tid
//   offset    = linearIdx * localDimSize
//
// Input:
//   tensor<4096xf32>, grid = 4x2, grid_axes = [0, 1], slice_axis = 0
//
// Output:
//   tensor<512xf32>, offset = (cid * ntpc + tid) * 512

shard.grid @nsp(shape = 4x2)

func.func @kernel_flattened(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                            %arg2: i32, %arg3: i32, %arg4: i32,
                            %arg5: i32, %arg6: i32, %arg7: i32)
    -> tensor<512xf32> {
  %rcast = memref.reinterpret_cast %arg0 to
      offset: [0], sizes: [4096], strides: [1]
      : memref<?xf32> to memref<4096xf32, strided<[1]>>

  %input = bufferization.to_tensor %rcast restrict
      : memref<4096xf32, strided<[1]>> to tensor<4096xf32>

  %local = shard.all_slice %input on @nsp grid_axes = [0, 1] slice_axis = 0
      : tensor<4096xf32> -> tensor<512xf32>

  return %local : tensor<512xf32>
}

// CHECK-LABEL: func.func @kernel_flattened(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: memref<?xf32>
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: memref<?xf32>
// CHECK-SAME: %[[NTPC_ARG:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %[[TID_ARG:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: %[[CID_ARG:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32

// CHECK-DAG: %[[C512:[A-Za-z0-9_]+]] = arith.constant 512 : index

// CHECK: %[[BASE:[A-Za-z0-9_]+]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [4096], strides: [1]
// CHECK: %[[TENSOR:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[BASE]] restrict

// grid_axes = [0, 1] means flattened participant index:
//   cid * ntpc + tid
// CHECK-DAG: %[[CID:[A-Za-z0-9_]+]] = arith.index_cast %[[CID_ARG]] : i32 to index
// CHECK-DAG: %[[TID:[A-Za-z0-9_]+]] = arith.index_cast %[[TID_ARG]] : i32 to index
// CHECK-DAG: %[[NTPC:[A-Za-z0-9_]+]] = arith.index_cast %[[NTPC_ARG]] : i32 to index
// CHECK: %[[MUL:[A-Za-z0-9_]+]] = arith.muli %[[CID]], %[[NTPC]] : index
// CHECK: %[[LINEAR_IDX:[A-Za-z0-9_]+]] = arith.addi %[[MUL]], %[[TID]] : index
// CHECK: %[[OFFSET:[A-Za-z0-9_]+]] = arith.muli %[[LINEAR_IDX]], %[[C512]] : index

// CHECK: %[[SLICE:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TENSOR]][%[[OFFSET]]] [512] [1] : tensor<4096xf32> to tensor<512xf32>
// CHECK: return %[[SLICE]] : tensor<512xf32>

// CHECK-NOT: shard.all_slice
// CHECK-NOT: shard.grid
