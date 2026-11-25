//===- broadcasted-reduction-to-butterfly.mlir ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -broadcasted-reduction-to-butterfly | FileCheck %s

// -----
// Reduction's result feeds vector.broadcast directly (no scalar chain).
// vector<4xf32> => 2-step butterfly with XOR strides 2 then 1.
// The reduction's accumulator (init) is folded lane-wise via splat + addf.
// CHECK-LABEL: func.func @reduce_add_direct_broadcast
// CHECK-SAME:    (%[[V:.*]]: vector<4xf32>) -> vector<4xf32>
// 2-step butterfly at source width.
// CHECK:         %[[T0:.*]] = vector.shuffle %[[V]], %[[V]] [2, 3, 0, 1] : vector<4xf32>, vector<4xf32>
// CHECK:         %[[B0:.*]] = arith.addf %[[V]], %[[T0]]
// CHECK:         %[[T1:.*]] = vector.shuffle %[[B0]], %[[B0]] [1, 0, 3, 2] : vector<4xf32>, vector<4xf32>
// CHECK:         %[[B1:.*]] = arith.addf %[[B0]], %[[T1]]
// Replicate 4→8→16→32 to reach native HVX width.
// CHECK:         %[[R8:.*]] = vector.shuffle %[[B1]], %[[B1]] [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
// CHECK:         %[[R16:.*]] = vector.shuffle %[[R8]], %[[R8]] [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
// CHECK:         %[[R32:.*]] = vector.shuffle %[[R16]], %[[R16]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// Splat acc to native width, fold, then contract back to broadcast width.
// CHECK:         %[[ACC_V:.*]] = vector.broadcast %{{.*}} : f32 to vector<32xf32>
// CHECK:         %[[SUM:.*]] = arith.addf %[[R32]], %[[ACC_V]]
// CHECK:         %[[OUT:.*]] = vector.extract_strided_slice %[[SUM]] {offsets = [0], sizes = [4], strides = [1]} : vector<32xf32> to vector<4xf32>
// CHECK:         return %[[OUT]]
// CHECK-NOT:     vector.reduction
func.func @reduce_add_direct_broadcast(%v: vector<4xf32>) -> vector<4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<4xf32> into f32
  %bcast = vector.broadcast %sum : f32 to vector<4xf32>
  return %bcast : vector<4xf32>
}

// -----
// Softmax-style chain: reduction -> arith.divf -> vector.broadcast.
// Chain is vectorized lane-wise; the non-chain divf operand (1.0) is splatted.
// CHECK-LABEL: func.func @reduce_add_divf_chain
// CHECK-SAME:    (%[[V:.*]]: vector<4xf32>) -> vector<4xf32>
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [2, 3, 0, 1]
// CHECK:         arith.addf
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [1, 0, 3, 2]
// CHECK:         arith.addf
// Replicate 4→8→16→32 to reach native HVX width.
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 0, 1, 2, 3]
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Acc fold and chain op at native width; contract back to broadcast width.
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         %[[INV:.*]] = arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         vector.extract_strided_slice %[[INV]] {offsets = [0], sizes = [4], strides = [1]} : vector<32xf32> to vector<4xf32>
// CHECK:         return
// CHECK-NOT:     vector.reduction
// CHECK-NOT:     arith.divf %{{.*}}, %{{.*}} : f32
func.func @reduce_add_divf_chain(%v: vector<4xf32>) -> vector<4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<4xf32> into f32
  %inv = arith.divf %one, %sum : f32
  %bcast = vector.broadcast %inv : f32 to vector<4xf32>
  return %bcast : vector<4xf32>
}

// -----
// RMSNorm-style chain mixing arith and math ops: reduction -> divf -> addf ->
// math.sqrt -> divf -> broadcast. Each chain op (including math.sqrt) is
// rewritten as its lane-wise vector equivalent.
// CHECK-LABEL: func.func @reduce_add_math_chain
// CHECK-SAME:    (%[[V:.*]]: vector<4xf32>) -> vector<4xf32>
// CHECK:         vector.shuffle
// CHECK:         arith.addf
// CHECK:         vector.shuffle
// CHECK:         arith.addf
// Replicate 4→8→16→32.
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 0, 1, 2, 3]
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Chain runs at native width.
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         math.sqrt %{{.*}} : vector<32xf32>
// CHECK:         arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// Contract to broadcast width.
// CHECK:         vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [4], strides = [1]} : vector<32xf32> to vector<4xf32>
// CHECK:         return
// CHECK-NOT:     vector.reduction
// CHECK-NOT:     math.sqrt %{{.*}} : f32
func.func @reduce_add_math_chain(%v: vector<4xf32>) -> vector<4xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_n = arith.constant 4.000000e+00 : f32
  %cst_eps = arith.constant 1.000000e-06 : f32
  %one = arith.constant 1.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst_0 : vector<4xf32> into f32
  %mean = arith.divf %sum, %cst_n : f32
  %var = arith.addf %mean, %cst_eps : f32
  %rms = math.sqrt %var : f32
  %inv = arith.divf %one, %rms : f32
  %bcast = vector.broadcast %inv : f32 to vector<4xf32>
  return %bcast : vector<4xf32>
}

// -----
// Different combining kind: max reduction with direct broadcast.
// Verifies the pass dispatches to arith.maxnumf in the butterfly.
// CHECK-LABEL: func.func @reduce_max_direct_broadcast
// CHECK-SAME:    (%[[V:.*]]: vector<4xf32>) -> vector<4xf32>
// 2-step butterfly at source width.
// CHECK:         vector.shuffle %[[V]], %[[V]] [2, 3, 0, 1]
// CHECK:         arith.maxnumf
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [1, 0, 3, 2]
// CHECK:         %[[B1:.*]] = arith.maxnumf
// Replicate 4→8→16→32.
// CHECK:         vector.shuffle %[[B1]], %[[B1]] [0, 1, 2, 3, 0, 1, 2, 3]
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
// CHECK:         %[[R32:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Fold acc at native width, contract back.
// CHECK:         %[[M:.*]] = arith.maxnumf %[[R32]], %{{.*}} : vector<32xf32>
// CHECK:         vector.extract_strided_slice %[[M]] {offsets = [0], sizes = [4], strides = [1]} : vector<32xf32> to vector<4xf32>
// CHECK:         return
// CHECK-NOT:     vector.reduction
func.func @reduce_max_direct_broadcast(%v: vector<4xf32>) -> vector<4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %m = vector.reduction <maxnumf>, %v, %neg_inf : vector<4xf32> into f32
  %bcast = vector.broadcast %m : f32 to vector<4xf32>
  return %bcast : vector<4xf32>
}

// -----
// Negative: the reduction's scalar result has more than one user, so the chain
// "branches". The pass requires single-use chain ops and must leave the IR
// untouched.
// CHECK-LABEL: func.func @neg_chain_branches
// CHECK:         vector.reduction <add>
// CHECK:         vector.broadcast %{{.*}} : f32 to vector<4xf32>
// CHECK-NOT:     vector.shuffle
func.func @neg_chain_branches(%v: vector<4xf32>) -> (vector<4xf32>, f32) {
  %cst = arith.constant 0.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<4xf32> into f32
  %bcast = vector.broadcast %sum : f32 to vector<4xf32>
  return %bcast, %sum : vector<4xf32>, f32
}

// -----
// Negative: an op outside the arith/math dialects sits in the chain (here a
// `func.call`), so the chain whitelist rejects the match.
// CHECK-LABEL: func.func @neg_unsupported_dialect_in_chain
// CHECK:         vector.reduction <add>
// The asm printer elides the `func.` dialect prefix for `func.call`.
// CHECK:         call @scalar_extern
// CHECK:         vector.broadcast %{{.*}} : f32 to vector<4xf32>
// CHECK-NOT:     vector.shuffle
func.func private @scalar_extern(f32) -> f32

func.func @neg_unsupported_dialect_in_chain(%v: vector<4xf32>) -> vector<4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<4xf32> into f32
  %t = func.call @scalar_extern(%sum) : (f32) -> f32
  %bcast = vector.broadcast %t : f32 to vector<4xf32>
  return %bcast : vector<4xf32>
}

// -----
// Negative: source vector length is not a power of two; the butterfly cannot
// be built and the pass leaves the IR untouched.
// CHECK-LABEL: func.func @neg_non_power_of_two
// CHECK:         vector.reduction <add>
// CHECK:         vector.broadcast %{{.*}} : f32 to vector<3xf32>
// CHECK-NOT:     vector.shuffle
func.func @neg_non_power_of_two(%v: vector<3xf32>) -> vector<3xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<3xf32> into f32
  %bcast = vector.broadcast %sum : f32 to vector<3xf32>
  return %bcast : vector<3xf32>
}

// -----
// Source vector exceeds the native HVX width (vector<64xf32> = 256B). The
// butterfly stays at the source width, but immediately after we drop down to
// vector<32xf32> via vector.extract_strided_slice and run the scalar chain
// (here a divf) at 32xf32. Because the broadcast consumer wants vector<64xf32>
// we expand back with one vector.shuffle whose mask replicates lanes 0..31.
// CHECK-LABEL: func.func @reduce_add_wide_src_chain_at_native
// CHECK-SAME:    (%[[V:.*]]: vector<64xf32>) -> vector<64xf32>
// 6 butterfly steps at vector<64xf32>; spot-check the first.
// CHECK:         vector.shuffle %[[V]], %[[V]] {{.*}} : vector<64xf32>, vector<64xf32>
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<64xf32>
// Drop to native width before the chain.
// CHECK:         %[[N32:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [32], strides = [1]} : vector<64xf32> to vector<32xf32>
// The 1.0 splat folds from vector.broadcast into a vector constant.
// CHECK:         %[[INV:.*]] = arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// Replicate 32->64 with a single shuffle whose mask is [0..31, 0..31].
// CHECK:         %[[OUT:.*]] = vector.shuffle %[[INV]], %[[INV]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
// CHECK:         return %[[OUT]]
// CHECK-NOT:     arith.divf %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK-NOT:     arith.divf %{{.*}}, %{{.*}} : f32
// CHECK-NOT:     vector.reduction
func.func @reduce_add_wide_src_chain_at_native(%v: vector<64xf32>) -> vector<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<64xf32> into f32
  %inv = arith.divf %one, %sum : f32
  %bcast = vector.broadcast %inv : f32 to vector<64xf32>
  return %bcast : vector<64xf32>
}

// -----
// Source vector at native width (vector<32xf32>) but broadcast consumer is
// wider (vector<64xf32>). Today's pass would reject this (bcastTy != srcVecTy);
// the new behaviour is butterfly + chain at vector<32xf32>, then a single
// replicate-shuffle 32->64 to satisfy the consumer.
// CHECK-LABEL: func.func @reduce_add_native_src_wide_bcast
// CHECK-SAME:    (%[[V:.*]]: vector<32xf32>) -> vector<64xf32>
// CHECK:         vector.shuffle %[[V]], %[[V]] {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK-NOT:     vector.extract_strided_slice
// CHECK:         arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         %[[OUT:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
// CHECK:         return %[[OUT]]
// CHECK-NOT:     vector.reduction
func.func @reduce_add_native_src_wide_bcast(%v: vector<32xf32>) -> vector<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<32xf32> into f32
  %inv = arith.divf %one, %sum : f32
  %bcast = vector.broadcast %inv : f32 to vector<64xf32>
  return %bcast : vector<64xf32>
}

// -----
// Broadcast consumer is 4x wider than native (vector<128xf32>). The expand
// uses two chained vector.shuffle ops: 32->64, then 64->128.
// CHECK-LABEL: func.func @reduce_add_native_src_4x_bcast
// CHECK-SAME:    (%[[V:.*]]: vector<32xf32>) -> vector<128xf32>
// CHECK:         arith.divf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK:         vector.shuffle %{{.*}}, %{{.*}} {{\[}}0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31{{\]}} : vector<32xf32>, vector<32xf32>
// CHECK:         %[[OUT:.*]] = vector.shuffle %{{.*}}, %{{.*}} {{\[}}0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63{{\]}} : vector<64xf32>, vector<64xf32>
// CHECK:         return %[[OUT]]
// CHECK-NOT:     vector.reduction
func.func @reduce_add_native_src_4x_bcast(%v: vector<32xf32>) -> vector<128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %sum = vector.reduction <add>, %v, %cst : vector<32xf32> into f32
  %inv = arith.divf %one, %sum : f32
  %bcast = vector.broadcast %inv : f32 to vector<128xf32>
  return %bcast : vector<128xf32>
}

// -----
// f16 source already at native width (vector<64xf16> = 128B). chainLen ==
// srcLen == bcastLen == 64, so neither extract_strided_slice nor an expand
// shuffle should appear; the chain runs at vector<64xf16> as before.
// CHECK-LABEL: func.func @reduce_max_f16_native
// CHECK-SAME:    (%[[V:.*]]: vector<64xf16>) -> vector<64xf16>
// CHECK:         vector.shuffle %[[V]], %[[V]] {{.*}} : vector<64xf16>, vector<64xf16>
// CHECK:         arith.maxnumf
// CHECK:         return
// CHECK-NOT:     vector.extract_strided_slice
// CHECK-NOT:     vector.reduction
func.func @reduce_max_f16_native(%v: vector<64xf16>) -> vector<64xf16> {
  %neg_inf = arith.constant 0xFC00 : f16
  %m = vector.reduction <maxnumf>, %v, %neg_inf : vector<64xf16> into f16
  %bcast = vector.broadcast %m : f16 to vector<64xf16>
  return %bcast : vector<64xf16>
}
