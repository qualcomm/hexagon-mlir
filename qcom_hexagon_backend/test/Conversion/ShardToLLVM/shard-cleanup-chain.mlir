//===- shard-cleanup-chain.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// This test validates that shard.shard annotations are dropped before
// shard.sharding is erased, even when multiple shard.shard users share the same
// !shard.sharding value.

module {
  shard.grid @nsp(shape = 16)

  func.func @drop_shard_annotations(%a: tensor<128xf32>) -> tensor<128xf32> {
    %sh = shard.sharding @nsp split_axes = [[0]]
      : !shard.sharding

    %a0 = shard.shard %a to %sh annotate_for_users
      : tensor<128xf32>

    %a1 = shard.shard %a0 to %sh annotate_for_users
      : tensor<128xf32>

    return %a1 : tensor<128xf32>
  }
}

// CHECK-LABEL: func.func @drop_shard_annotations(
// CHECK: return %arg0 : tensor<128xf32>

// CHECK-NOT: shard.shard
// CHECK-NOT: shard.sharding
// CHECK-NOT: shard.grid
