//===- crouton-addr-strided.mlir - tile-grid strides come from the layout ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An `hmx.mma` operand can be a `memref.subview` of a larger crouton array --
// which is how the whole-weight residency (B2) hands each program its N block.
// The crouton address arithmetic uses the operand's *own* tile-grid stride
// (`stride(dim) / 1024` croutons), not the grid's `dimSize`; a subview's tile is
// separated from its neighbour by the *whole* array's stride.
//
// Under layout A the weight grid is [Nt, Kt] (Wᵀ), so a B2 N block is a slice on
// dim0 (N) whose stride is the whole `Kt`. For a dense slice that stride equals
// the subview's `dimSize(1)`, so the strided and dense emissions coincide here
// (both 4 croutons for the case below).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// Whole weight crouton array [8, 4, 16, 32, 2] (Wᵀ: dim0 = N, dim1 = K) sliced
// to the N block at offset 2: the subview is [2, 4, ...], and its dim-0 stride
// is 4096 elements (4 croutons), so the tile row stride constant is 4.
// TODO(layout-A): a dense N-slice has stride(0)/1024 == dimSize(1) (both are
// Kt), so this case no longer distinguishes the operand's stride from its
// width; the reviewer should confirm whether it needs a K-slice (or another
// non-dense layout) to keep exercising `croutonTileStride`.
// CHECK-LABEL: func.func @mma_strided_wt
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// CHECK: %[[WTS:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[WTN:.*]] = llvm.trunc
// CHECK: %[[NROW:.*]] = llvm.mul %[[WTN]], %[[WTS]]
// CHECK: %[[WTK:.*]] = llvm.trunc
// CHECK: %[[NK:.*]] = llvm.add %[[NROW]], %[[WTK]]
// CHECK: %[[WTB:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[WTILE:.*]] = llvm.mul %[[NK]], %[[WTB]]
// CHECK: llvm.call @hmx_mma_f16(
func.func @mma_strided_wt(%bias: memref<256xi8, 1>,
                          %act: memref<4x4x16x32x2xf16, 1>,
                          %whole: memref<8x4x16x32x2xf16, 1>,
                          %ar: memref<4x2x16x32x2xf16, 1>,
                          %m: index, %n: index, %k: index) {
  %wsub = memref.subview %whole[2, 0, 0, 0, 0] [2, 4, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<8x4x16x32x2xf16, 1> to memref<2x4x16x32x2xf16, strided<[4096, 1024, 64, 2, 1], offset: 8192>, 1>
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wsub, %m, %n, %k {n_croutons = 1 : i32}
      : memref<4x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, strided<[4096, 1024, 64, 2, 1], offset: 8192>, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// -----

// Dense weight [3, 4, 16, 32, 2] (Wᵀ: dim0 = N, dim1 = K): stride(0)/1024 ==
// dimSize(1) == 4, and the column step is 1, so the lowering emits the dense
// form verbatim.
// CHECK-LABEL: func.func @mma_dense_wt
// CHECK: %[[DTS:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[DTN:.*]] = llvm.trunc
// CHECK: %[[DROW:.*]] = llvm.mul %[[DTN]], %[[DTS]]
// CHECK: %[[DTK:.*]] = llvm.trunc
// CHECK: %[[DKN:.*]] = llvm.add %[[DROW]], %[[DTK]]
// CHECK: %[[DTB:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[DTILE:.*]] = llvm.mul %[[DKN]], %[[DTB]]
// CHECK: llvm.call @hmx_mma_f16(
func.func @mma_dense_wt(%bias: memref<256xi8, 1>,
                        %act: memref<4x4x16x32x2xf16, 1>,
                        %wt: memref<3x4x16x32x2xf16, 1>,
                        %ar: memref<4x2x16x32x2xf16, 1>,
                        %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32}
      : memref<4x4x16x32x2xf16, 1>, memref<3x4x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}
