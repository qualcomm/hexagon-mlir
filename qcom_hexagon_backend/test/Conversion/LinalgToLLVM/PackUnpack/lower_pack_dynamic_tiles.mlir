//===- lower_pack_dynamic_tiles.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-pack))' -verify-diagnostics

// lowerPack does not support dynamic inner tiles; the pass reports an error.
func.func @dynamic_inner_tiles(%src: tensor<64x64xf32>, %dst: tensor<?x?x?x?xf32>, %t: index) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{failed to lower linalg.pack}}
  %0 = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [%t, %t] into %dst : tensor<64x64xf32> -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
