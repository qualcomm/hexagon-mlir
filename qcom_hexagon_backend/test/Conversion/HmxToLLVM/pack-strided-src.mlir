//===- pack-strided-src.mlir - pack uses the source's real row stride -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The row-major source of a pack can be one tile of a wider matrix. The
// diagnosed bug: the leaf used the *logical width* (`cols`/`n`) as the row
// stride, so an N/K-split source -- here `memref<2048x256xf16, strided<[512,
// 1]>>`, whose rows are 512 elements apart while it is only 256 columns wide --
// read the wrong element of every row past the first. The leaf now takes the
// real row stride (`src_stride`, the source's `stride(rank-2)`), the source-side
// twin of the unpack `dst_stride`: `src_stride == cols` for a dense source, and
// the wider row stride for a strided view. `cols`/`n` remain the column count
// used only for the zero-fill boundary.
//
// This file pins that contract on the lowering: the strided source must pass
// the wider row stride, the dense source must pass the width itself.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// Strided source: rows=2048, cols=256, row stride=512. `cols` (the 4th
// argument) must stay 256, and `src_stride` (the 5th) must be 512 -- not 256.
// CHECK-LABEL: func.func @pack_weight_strided
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// CHECK: %[[N:.*]] = llvm.mlir.constant(256 : i32)
// CHECK: %[[S:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_pack_weight_f16({{.*}}, {{.*}}, {{.*}}, %[[N]], %[[S]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @pack_weight_strided(%wsrc: memref<2048x256xf16, strided<[512, 1]>>,
                               %wt: memref<8x64x16x32x2xf16, 1>,
                               %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<2048x256xf16, strided<[512, 1]>>)
      outs(%wt : memref<8x64x16x32x2xf16, 1>)
  return
}

// -----

// The same strided source on the activation side: cols=256, src_stride=512.
// CHECK-LABEL: func.func @pack_act_strided
// CHECK: %[[N:.*]] = llvm.mlir.constant(256 : i32)
// CHECK: %[[S:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_pack_act_f16({{.*}}, {{.*}}, {{.*}}, %[[N]], %[[S]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_act_strided(%src: memref<2048x256xf16, strided<[512, 1]>>,
                            %act: memref<64x8x16x32x2xf16, 1>,
                            %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<2048x256xf16, strided<[512, 1]>>)
      outs(%act : memref<64x8x16x32x2xf16, 1>)
  return
}

// -----

// Dense source (`memref<64x64xf16>`): stride(0) == cols, so the lowering reuses
// the width value and the 4th and 5th arguments are the same SSA name.
// CHECK-LABEL: func.func @pack_weight_dense
// CHECK: llvm.call @hmx_pack_weight_f16({{.*}}, {{.*}}, {{.*}}, %[[C:.*]], %[[C]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_weight_dense(%wsrc: memref<64x64xf16>,
                             %wt: memref<2x2x16x32x2xf16, 1>,
                             %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x64xf16>)
      outs(%wt : memref<2x2x16x32x2xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @pack_act_dense
// CHECK: llvm.call @hmx_pack_act_f16({{.*}}, {{.*}}, {{.*}}, %[[C:.*]], %[[C]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_act_dense(%src: memref<64x64xf16>,
                          %act: memref<2x2x16x32x2xf16, 1>,
                          %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  return
}
