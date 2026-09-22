//===- unpack-f32-strided-residual.mlir - strided f32 tail strides ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The fused f32 tail (`hmx.unpack_acc_f32`) writes a row-major block whose row
// stride is `stride(rank-2)`, not the block width. A strided destination or
// residual (an N-tile view of a wider matrix) must be passed the real stride:
// using the width walks the wrong rows -- the same class as the pack
// `src_stride` and the f16 unpack `dst_stride` fixes.
//
// Here dst and residual are `memref<64x32xf32, strided<[96, 1]>>`: width 32,
// row stride 96. Both the dst stride and the residual stride must be 96.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_residual_strided
// The residual stride constant is emitted before the destination stride, so the
// two `96` constants are the residual and the destination row strides (both
// stride(rank-2) = 96), not the width (32).
// CHECK: llvm.mlir.constant(96 : i32)
// CHECK: llvm.mlir.constant(96 : i32)
// CHECK: llvm.call @hmx_unpack_acc_f32({{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @tail_residual_strided(%ar: memref<2x1x16x32x2xf16, 1>,
                                 %dst: memref<64x32xf32, strided<[96, 1]>>,
                                 %res: memref<64x32xf32, strided<[96, 1]>>,
                                 %row: index, %col: index) {
  hmx.unpack_acc_f32 ins(%ar, %row, %col, %res
                         : memref<2x1x16x32x2xf16, 1>,
                           memref<64x32xf32, strided<[96, 1]>>)
      outs(%dst : memref<64x32xf32, strided<[96, 1]>>)
  return
}
