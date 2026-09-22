//===- verify.mlir - Hmx dialect verifier negative cases ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

//--- !!! the crouton arrays must live in VTCM (memory space 1)
func.func @bad_space(%act: memref<2x4x16x32x2xf16>, %wt: memref<2x4x16x32x2xf16>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{must be in VTCM}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32} : memref<2x4x16x32x2xf16>, memref<2x4x16x32x2xf16>
  return
}

//--- !!! ... and be a 2D grid of croutons, not a single crouton
func.func @bad_grid(%act: memref<16x32x2xf16, 1>, %wt: memref<2x4x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{must be a 2D grid of croutons}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32} : memref<16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  return
}

//--- !!! the K tile counts must agree. The weight grid is [Nt, Kt] (stored Wᵀ),
//--- so K is dim1 on BOTH operands; here the weight's dim1 = 8 != act's 4.
func.func @bad_ktiles(%act: memref<2x4x16x32x2xf16, 1>, %wt: memref<2x8x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{the K tile counts must agree}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32} : memref<2x4x16x32x2xf16, 1>, memref<2x8x16x32x2xf16, 1>
  return
}

//--- !!! n_croutons must be at least 1
func.func @bad_ncroutons(%act: memref<2x4x16x32x2xf16, 1>, %wt: memref<2x4x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{n_croutons must be at least 1}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 0 : i32} : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  return
}

//--- !!! ... and at most 32 (the engine's five-bit Rt[dC])
func.func @too_many_ncroutons(%act: memref<2x4x16x32x2xf16, 1>, %wt: memref<2x4x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{n_croutons must be at most 32}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 33 : i32} : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  return
}

//--- !!! the conversion state block is 256 B
func.func @bad_conv_state(%bias: memref<128xi8, 1>, %ar: memref<2x2x16x32x2xf16, 1>, %m: index, %n: index) {
  // expected-error @+1 {{must be the 256-byte conversion state}}
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 0 : i32} : memref<128xi8, 1>, memref<2x2x16x32x2xf16, 1>
  return
}

//--- !!! bias_set selects one of four register sets
func.func @bad_bias_set(%bias: memref<256xi8, 1>, %ar: memref<2x2x16x32x2xf16, 1>, %m: index, %n: index) {
  // expected-error @+1 {{bias_set must be in [0, 3]}}
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 7 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
  return
}

//--- !!! hmx.matmul operates on croutons: [Mt, Kt, 16, 32, 2]
func.func @bad_matmul_not_crouton(%a: tensor<2x4x32x32x2xf16>, %b: tensor<2x4x16x32x2xf16>, %c: tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16> {
  // expected-error @+1 {{expects croutons}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x4x32x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%c : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
  return %0 : tensor<2x2x16x32x2xf16>
}

//--- !!! the tile counts must form a product: rhs is [Nt, Kt], so K is dim1
func.func @bad_matmul_inner(%a: tensor<2x4x16x32x2xf16>, %b: tensor<2x8x16x32x2xf16>, %c: tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16> {
  // expected-error @+1 {{inner tile counts must agree}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x4x16x32x2xf16>, tensor<2x8x16x32x2xf16>) outs(%c : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
  return %0 : tensor<2x2x16x32x2xf16>
}

//--- !!! the weight is stored as Wᵀ: a `rhs` still laid out as the old
//--- [Kt, Nt] is rejected, because K is now the grid's second dim. Here the old
//--- layout was valid (rhs[0] == lhs Kt), but rhs[1] = 2 != lhs Kt = 4.
func.func @bad_matmul_rhs_old_layout(%a: tensor<2x4x16x32x2xf16>, %b: tensor<4x2x16x32x2xf16>, %c: tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16> {
  // expected-error @+1 {{inner tile counts must agree}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x4x16x32x2xf16>, tensor<4x2x16x32x2xf16>) outs(%c : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
  return %0 : tensor<2x2x16x32x2xf16>
}

//--- !!! output tile count
func.func @bad_matmul_out(%a: tensor<2x4x16x32x2xf16>, %b: tensor<2x4x16x32x2xf16>, %c: tensor<2x3x16x32x2xf16>) -> tensor<2x3x16x32x2xf16> {
  // expected-error @+1 {{output tile count must be [Mt, Nt]}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%c : tensor<2x3x16x32x2xf16>) -> tensor<2x3x16x32x2xf16>
  return %0 : tensor<2x3x16x32x2xf16>
}

//--- !!! a pack range may not exceed the array's contiguous (K) axis
func.func @too_many_pack_act(%src: memref<64x64xf16>, %act: memref<2x2x16x32x2xf16, 1>, %row: index, %col: index) {
  // expected-error @+1 {{count 3 exceeds the K tile extent 2}}
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>) {count = 3 : i64}
  return
}

//--- !!! ... and an unpack range the 16 row-pairs of one tile row
func.func @too_many_unpack(%ar: memref<2x1x16x32x2xf16, 1>, %dst16: memref<64x32xf16>, %row: index, %col: index) {
  // expected-error @+1 {{count 17 exceeds the row-pair extent 16}}
  hmx.unpack_acc ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>) outs(%dst16 : memref<64x32xf16>) {count = 17 : i64}
  return
}

//--- !!! ... the fused tail shares the same row-pair bound
func.func @too_many_unpack_f32(%ar: memref<2x1x16x32x2xf16, 1>, %dst32: memref<64x32xf32>, %row: index, %col: index) {
  // expected-error @+1 {{count 17 exceeds the row-pair extent 16}}
  hmx.unpack_acc_f32 ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>) outs(%dst32 : memref<64x32xf32>) {count = 17 : i64}
  return
}

//--- !!! hmx.stage stages a static rank-2 f16/f32 row-major source (its row
//--- stride is threaded to the DMA, so the shape must be known)
func.func @bad_stage_src(%src: memref<?x1024xf16>, %slot: memref<32x1024xf16, 1>, %status: memref<1xi32>, %row: index) {
  // expected-error @+1 {{src must be a static rank-2 f16/f32 memref}}
  %tok = hmx.stage ins(%src, %row : memref<?x1024xf16>) outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  return
}

//--- !!! ... the slot is a VTCM buffer
func.func @bad_stage_slot_space(%src: memref<64x1024xf16>, %slot: memref<32x1024xf16>, %status: memref<1xi32>, %row: index) {
  // expected-error @+1 {{dst must be in VTCM}}
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<32x1024xf16>, memref<1xi32>) -> i32
  return
}

//--- !!! ... exactly one crouton tile tall
func.func @bad_stage_slot_height(%src: memref<64x1024xf16>, %slot: memref<16x1024xf16, 1>, %status: memref<1xi32>, %row: index) {
  // expected-error @+1 {{dst must be one crouton tile tall}}
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<16x1024xf16, 1>, memref<1xi32>) -> i32
  return
}

//--- !!! ... and as wide as the source
func.func @bad_stage_slot_width(%src: memref<64x1024xf16>, %slot: memref<32x512xf16, 1>, %status: memref<1xi32>, %row: index) {
  // expected-error @+1 {{dst columns (512) must match the source columns (1024)}}
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<32x512xf16, 1>, memref<1xi32>) -> i32
  return
}

//--- !!! the status word is exactly one i32
func.func @bad_stage_status(%src: memref<64x1024xf16>, %slot: memref<32x1024xf16, 1>, %status: memref<2xi32>, %row: index) {
  // expected-error @+1 {{status must be one i32 word: memref<1xi32>}}
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<32x1024xf16, 1>, memref<2xi32>) -> i32
  return
}

//--- !!! hmx.await waits on a VTCM slot
func.func @bad_await_space(%slot: memref<32x1024xf16>, %tok: i32) {
  // expected-error @+1 {{dst must be in VTCM}}
  %r = hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16>) -> memref<32x1024xf16>
  return
}

//--- !!! ... and returns exactly that slot (the result is the value edge the
//--- compute side consumes)
func.func @bad_await_result(%slot: memref<32x1024xf16, 1>, %tok: i32) {
  // expected-error @+1 {{must match dst type}}
  %r = hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>) -> memref<16x1024xf16, 1>
  return
}
