//===- lower-process-linear-index.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// This test validates the ShardToLLVM lowering of shard.process_linear_index.
// The Hexagon entry-point ABI is interpreted from the tail of the function
// signature as:
//   [..., ntpc, num_cores, reserved0, tid, cid, reserved1]
// Therefore, for a function with exactly six ABI arguments:
//   ntpc = %arg0
//   tid  = %arg3
//   cid  = %arg4
// shard.process_linear_index must lower to:
//   linearIdx = cid * ntpc + tid

module {
  shard.grid @nsp(shape = 16)

  func.func @lower_process_linear_index(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> index {
    %idx = shard.process_linear_index on @nsp : index
    return %idx : index
  }
}

// CHECK-LABEL: func.func @lower_process_linear_index(
// CHECK-SAME: %[[NTPC_ARG:[^:]*]]: i32,
// CHECK-SAME: %{{[^:]*}}: i32,
// CHECK-SAME: %{{[^:]*}}: i32,
// CHECK-SAME: %[[TID_ARG:[^:]*]]: i32,
// CHECK-SAME: %[[CID_ARG:[^:]*]]: i32,
// CHECK-SAME: %{{[^:]*}}: i32)

// CHECK-DAG: %[[CID:.*]] = arith.index_cast %[[CID_ARG]] : i32 to index
// CHECK-DAG: %[[TID:.*]] = arith.index_cast %[[TID_ARG]] : i32 to index
// CHECK-DAG: %[[NTPC:.*]] = arith.index_cast %[[NTPC_ARG]] : i32 to index

// CHECK: %[[MUL:.*]] = arith.muli %[[CID]], %[[NTPC]] : index
// CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[TID]] : index
// CHECK: return %[[ADD]] : index

// CHECK-NOT: shard.process_linear_index
// CHECK-NOT: shard.grid
