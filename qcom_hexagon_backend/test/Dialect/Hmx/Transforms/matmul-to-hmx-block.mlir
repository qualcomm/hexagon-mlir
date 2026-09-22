//===- matmul-to-hmx-block.mlir - a bridge too large whole is walked in M ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// When the whole contraction's crouton arrays do not fit the VTCM budget but a
// block does, `matmul-to-hmx` walks M in blocks instead of refusing the op: one
// `hmx.matmul` per block on block-sized croutons, the whole weight, and each
// block written straight back into the carried row-major output. `blockM` is a
// divisor of M so the blocks tile M exactly (no partial block, no guard).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=20000}))' -split-input-file | FileCheck %s --check-prefix=SMALL
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=40000}))' -split-input-file | FileCheck %s --check-prefix=DIV
//===----------------------------------------------------------------------===//

// 64x64x64 on a 20000-byte budget: the whole bridge is 24576 bytes (over), the
// smallest block (one 32-row tile) is 16384 bytes (under), so M is walked in
// two 32-row blocks. The block's activation crouton is [1, 2] (32 rows, Kt=2),
// its read-out is [1, 2] (32 rows, Nt=2) and the weight stays whole at [2, 2].
//
// SMALL-LABEL: func.func @blocked_m
// The weight is packed once, whole, above the block loop.
// SMALL: hmx.pack_weight
// The block loop steps M by 32 (two iterations) and carries the row-major
// output; each iteration slices its 32 rows out of the activation.
// SMALL: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<64x64xf16>)
// SMALL: tensor.extract_slice %arg0[%arg2, %c0] [32, 64] [1, 1] : tensor<64x64xf16> to tensor<32x64xf16>
// SMALL: hmx.pack_act
// The block's croutons: activation [Mt=1, Kt=2], weight [Nt=2, Kt=2], read-out
// [Mt=1, Nt=2].
// SMALL: hmx.matmul ins({{.*}} : tensor<1x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>) outs({{.*}} : tensor<1x2x16x32x2xf16>) -> tensor<1x2x16x32x2xf16>
// SMALL: hmx.unpack_acc
// Each block is written back into the carried output.
// SMALL: tensor.insert_slice {{.*}} into %arg3[%arg2, %c0] [32, 64] [1, 1] : tensor<32x64xf16> into tensor<64x64xf16>
func.func @blocked_m(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// The M block is the largest divisor of the tile count that fits, not just one
// tile: 256x64x64 on a 40000-byte budget. The whole bridge is 73728 bytes; the
// largest fitting block is 96 rows, but 96/32 = 3 does not divide M/32 = 8, so
// the block drops to 2 tiles = 64 rows (24576 bytes) and M is walked in four
// 64-row blocks.
// DIV-LABEL: func.func @block_divisor
// DIV: tensor.extract_slice %arg0[%arg2, %c0] [64, 64] [1, 1] : tensor<256x64xf16> to tensor<64x64xf16>
// DIV: hmx.matmul ins({{.*}} : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>) outs({{.*}} : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
// DIV: tensor.insert_slice {{.*}} into %arg3[%arg2, %c0] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<256x64xf16>
func.func @block_divisor(%a: tensor<256x64xf16>, %b: tensor<64x64xf16>) -> tensor<256x64xf16> {
  %c = tensor.empty() : tensor<256x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<256x64xf16>, tensor<64x64xf16>)
                     outs(%c : tensor<256x64xf16>) -> tensor<256x64xf16>
  return %0 : tensor<256x64xf16>
}
