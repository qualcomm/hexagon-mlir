//===- lower-shard-all_slice.mlir -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// hard.all_slice lowers to tensor.extract_slice where the offset along
// the split axis is participantIndex * localDimSize.
//
// This case uses grid_axes = [0], so the participant index is the grid axis 0
// index, i.e. cid. It must not use the flattened linear index
// cid * ntpc + tid.
//
// Input:
//   tensor<2048xf32>, grid = 4x1, slice_axis = 0
//
// Output:
//   tensor<512xf32>, offset = cid * 512

shard.grid @nsp(shape = 4x1)

func.func @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                  %arg2: i32, %arg3: i32, %arg4: i32,
                  %arg5: i32, %arg6: i32, %arg7: i32)
    -> tensor<512xf32> {
  %rcast = memref.reinterpret_cast %arg0 to
      offset: [0], sizes: [2048], strides: [1]
      : memref<?xf32> to memref<2048xf32, strided<[1]>>

  %input = bufferization.to_tensor %rcast restrict
      : memref<2048xf32, strided<[1]>> to tensor<2048xf32>

  %local = shard.all_slice %input on @nsp grid_axes = [0] slice_axis = 0
      : tensor<2048xf32> -> tensor<512xf32>

  return %local : tensor<512xf32>
}

// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: memref<?xf32>
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: memref<?xf32>
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32
// CHECK-SAME: %[[CID_ARG:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: i32

// CHECK-DAG: %[[C512:[A-Za-z0-9_]+]] = arith.constant 512 : index

// The source memref view and tensor wrapper must be preserved.
// CHECK: %[[BASE:[A-Za-z0-9_]+]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [2048], strides: [1]
// CHECK: %[[TENSOR:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[BASE]] restrict

// grid_axes = [0] means axis-0 participant id only, i.e. cid.
// Do not expect cid * ntpc + tid here.
// CHECK: %[[CID:[A-Za-z0-9_]+]] = arith.index_cast %[[CID_ARG]] : i32 to index
// CHECK: %[[OFFSET:[A-Za-z0-9_]+]] = arith.muli %[[CID]], %[[C512]] : index

// CHECK: %[[SLICE:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TENSOR]][%[[OFFSET]]] [512] [1] : tensor<2048xf32> to tensor<512xf32>
// CHECK: return %[[SLICE]] : tensor<512xf32>

// CHECK-NOT: shard.all_slice
// CHECK-NOT: shard.grid
