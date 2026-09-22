//===- roundtrip.mlir - Hmx dialect roundtrip -----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @hmx_tile
func.func @hmx_tile(%bias: memref<256xi8, 1>, %act: memref<2x4x16x32x2xf16, 1>,
                    %wt: memref<2x4x16x32x2xf16, 1>,
                    %ar: memref<2x2x16x32x2xf16, 1>,
                    %m: index, %n: index, %k: index) {
  // CHECK: hmx.bias_init %{{.*}} : memref<256xi8, 1>
  hmx.bias_init %bias : memref<256xi8, 1>
  // CHECK: hmx.acc_clear
  hmx.acc_clear
  // CHECK: hmx.mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {n_croutons = 1 : i32} : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32}
      : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  // CHECK: hmx.acc_read %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {bias_set = 2 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
  return
}

// -----

// The crouton layout is a tensor encoding: `#hmx.crouton<logical = [M, N]>` on
// the rank-5 physical type. It roundtrips, and the logical shape is carried
// independently of the (redundant) grid so a verifier can cross-check the two.
// CHECK-LABEL: func.func @crouton_encoding
// CHECK-SAME: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
// CHECK-SAME: -> tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
func.func @crouton_encoding(%a: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
    ) -> tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>> {
  return %a : tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
}

// -----

// The same encoding is valid on any crouton geometry (here a single 32x32
// crouton), and an identity affine map is *not* the crouton layout: only the
// encoding names it.
// CHECK-LABEL: func.func @single_crouton
// CHECK-SAME: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
func.func @single_crouton(%a: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
    ) -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>> {
  return %a : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
}

// -----

// The fused tail roundtrips with and without its optional residual: absent is
// the plain unpack-to-f32, present is the unpack-widen-add in one op.
// CHECK-LABEL: func.func @fused_tail
func.func @fused_tail(%ar: memref<2x1x16x32x2xf16, 1>,
                      %dst32: memref<64x32xf32>,
                      %res: memref<64x32xf32>,
                      %row: index, %col: index) {
  // CHECK: hmx.unpack_acc_f32 ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<2x1x16x32x2xf16, 1>) outs(%{{.*}} : memref<64x32xf32>)
  hmx.unpack_acc_f32 ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst32 : memref<64x32xf32>)
  // CHECK: hmx.unpack_acc_f32 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<2x1x16x32x2xf16, 1>, memref<64x32xf32>) outs(%{{.*}} : memref<64x32xf32>)
  hmx.unpack_acc_f32 ins(%ar, %row, %col, %res : memref<2x1x16x32x2xf16, 1>, memref<64x32xf32>)
      outs(%dst32 : memref<64x32xf32>)
  return
}

// -----

// The optional `count` (the bulk range) roundtrips on all four layout ops.
// CHECK-LABEL: func.func @ranged_layout
func.func @ranged_layout(%src: memref<64x64xf16>, %wsrc: memref<64x96xf16>,
                         %act: memref<2x2x16x32x2xf16, 1>,
                         %wt: memref<3x2x16x32x2xf16, 1>,
                         %ar: memref<2x1x16x32x2xf16, 1>,
                         %dst16: memref<64x32xf16>, %dst32: memref<64x32xf32>,
                         %row: index, %col: index, %kt: index, %nt: index) {
  // CHECK: hmx.pack_act ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<64x64xf16>) outs(%{{.*}} : memref<2x2x16x32x2xf16, 1>) {count = 2 : i64}
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>) {count = 2 : i64}
  // CHECK: hmx.pack_weight ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<64x96xf16>) outs(%{{.*}} : memref<3x2x16x32x2xf16, 1>) {count = 2 : i64}
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x96xf16>)
      outs(%wt : memref<3x2x16x32x2xf16, 1>) {count = 2 : i64}
  // CHECK: hmx.unpack_acc ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<2x1x16x32x2xf16, 1>) outs(%{{.*}} : memref<64x32xf16>) {count = 8 : i64}
  hmx.unpack_acc ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst16 : memref<64x32xf16>) {count = 8 : i64}
  // CHECK: hmx.unpack_acc_f32 ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<2x1x16x32x2xf16, 1>) outs(%{{.*}} : memref<64x32xf32>) {count = 8 : i64}
  hmx.unpack_acc_f32 ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst32 : memref<64x32xf32>) {count = 8 : i64}
  return
}

// -----

// The staging pair roundtrips: `hmx.stage` takes the row-major source, the
// element row offset, the VTCM slot and the status word and returns the DMA
// token; `hmx.await` takes the token and the slot and returns the ready slot.
// CHECK-LABEL: func.func @stage_await
func.func @stage_await(%src: memref<64x1024xf16>, %slot: memref<32x1024xf16, 1>,
                       %status: memref<1xi32>, %row: index) {
  // CHECK: %[[TOK:.*]] = hmx.stage ins(%{{.*}}, %{{.*}} : memref<64x1024xf16>) outs(%{{.*}}, %{{.*}} : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  // CHECK: hmx.await ins(%[[TOK]] : i32) outs(%{{.*}} : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
  %ready = hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
  return
}
