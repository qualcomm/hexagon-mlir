//===- shard-to-llvm-all_slice-grid-axes.mlir -----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// This test validates that ShardToLLVM lowers shard.all_slice using the
// process index selected by grid_axes. Axis 0 uses cid, axis 1 uses tid, and
// the old flattened cid * ntpc + tid path is not used for these single-axis
// slices.

module {
  shard.grid @nsp(shape = 16x4)

  func.func @all_slice_grid_axis_0(
      %src: tensor<128x32xf32>,
      %ntpc: index,
      %num_cores: index,
      %reserved0: index,
      %tid: index,
      %cid: index,
      %reserved1: index) -> tensor<8x32xf32> {
    // CHECK-LABEL: func.func @all_slice_grid_axis_0
    // CHECK-DAG: %[[C8:[A-Za-z0-9_]+]] = arith.constant 8 : index
    // CHECK: %[[ROW_OFF:[A-Za-z0-9_]+]] = arith.muli %arg5, %[[C8]] : index
    // CHECK-NEXT: %[[SLICE:[A-Za-z0-9_]+]] = tensor.extract_slice %arg0[%[[ROW_OFF]], 0] [8, 32] [1, 1]
    // CHECK-NEXT: return %[[SLICE]] : tensor<8x32xf32>
    %0 = shard.all_slice %src on @nsp grid_axes = [0] slice_axis = 0
        : tensor<128x32xf32> -> tensor<8x32xf32>
    return %0 : tensor<8x32xf32>
  }

  func.func @all_slice_grid_axis_1(
      %src: tensor<32x64xf32>,
      %ntpc: index,
      %num_cores: index,
      %reserved0: index,
      %tid: index,
      %cid: index,
      %reserved1: index) -> tensor<32x16xf32> {
    // CHECK-LABEL: func.func @all_slice_grid_axis_1
    // CHECK-DAG: %[[C16:[A-Za-z0-9_]+]] = arith.constant 16 : index
    // CHECK: %[[COL_OFF:[A-Za-z0-9_]+]] = arith.muli %arg4, %[[C16]] : index
    // CHECK-NEXT: %[[SLICE:[A-Za-z0-9_]+]] = tensor.extract_slice %arg0[0, %[[COL_OFF]]] [32, 16] [1, 1]
    // CHECK-NEXT: return %[[SLICE]] : tensor<32x16xf32>
    %0 = shard.all_slice %src on @nsp grid_axes = [1] slice_axis = 1
        : tensor<32x64xf32> -> tensor<32x16xf32>
    return %0 : tensor<32x16xf32>
  }
}
