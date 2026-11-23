//===- fast_inverse.mlir --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hexagon-fast-inverse)' | FileCheck %s

// Unknown-sign denominator: apply runtime sign transfer (Quake rsqrt +
// bitcast/andi/ori sign fixup). The result is broadcast and multiplied.
// CHECK-LABEL:   func.func @test_scalar_division_f32(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<128xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<128xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_1]][] : memref<f32>
// CHECK:           arith.subi %c1597463007_i32,
// CHECK:           arith.andi
// CHECK:           arith.ori
// CHECK:           %[[VAL_5:.*]] = vector.broadcast %{{.*}} : f32 to vector<128xf32>
// CHECK:           %[[VAL_6:.*]] = arith.mulf %[[VAL_0]], %[[VAL_5]] : vector<128xf32>
// CHECK:           vector.transfer_write %[[VAL_6]], %[[VAL_2]]{{\[}}%[[VAL_3]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:           return
// CHECK:         }
// CHECK-NOT:       arith.divf
func.func @test_scalar_division_f32(%arg0: vector<128xf32>, %arg1: memref<f32>, %arg2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f32>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f32 to vector<128xf32>
  %div = arith.divf %arg0, %denominator : vector<128xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf32>, memref<128xf32>
  return
}

// This test converts the fp16 vector division with denominator as broadcast operation into
// a scalar div, and then broadcast the result followed by multiplication with numerator
// CHECK-LABEL:   func.func @test_scalar_division_f16(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<128xf16>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<f16>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<128xf16>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_1]][] : memref<f16>
// CHECK:           %[[VAL_5:.*]] = vector.broadcast %[[VAL_4]] : f16 to vector<128xf16>
// CHECK:           %[[VAL_6:.*]] = arith.divf %[[VAL_0]], %[[VAL_5]] : vector<128xf16>
// CHECK:           vector.transfer_write %[[VAL_6]], %[[VAL_2]]{{\[}}%[[VAL_3]]] {in_bounds = [true]} : vector<128xf16>, memref<128xf16>
// CHECK:           return
// CHECK:         }
func.func @test_scalar_division_f16(%arg0: vector<128xf16>, %arg1: memref<f16>, %arg2: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f16>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f16 to vector<128xf16>
  %div = arith.divf %arg0, %denominator : vector<128xf16>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf16>, memref<128xf16>
  return
}

// Unknown-sign denominator: apply runtime sign transfer (Quake rsqrt +
// bitcast/andi/ori sign fixup on the full vector).
// CHECK-LABEL:   func.func @test_fast_inverse_division_f32(
// CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:         arith.subi
// CHECK:         arith.andi
// CHECK:         arith.ori
// CHECK:         %[[VAL_4:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<256xf32>
// CHECK:         vector.transfer_write %[[VAL_4]], %[[VAL_2]][%[[VAL_3]]] {in_bounds = [true]} : vector<256xf32>, memref<256xf32>
// CHECK:         return
// CHECK:       }
// CHECK-NOT:   arith.divf
func.func @test_fast_inverse_division_f32(%arg0: vector<256xf32>, %arg1: vector<256xf32>, %arg2: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %div = arith.divf %arg0, %arg1 : vector<256xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<256xf32>, memref<256xf32>
  return
}


// This test ensures that the fp32 division of a constant 1.0f by sqrt(a)
// is transformed directly into the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_one_div_sqrt_f32(
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_one_div_sqrt_f32(%arg0: f32) -> f32 {
  %cst = arith.constant 1.000000e+00 : f32
  %1 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %cst, %1 fastmath<fast> : f32
  return %2 : f32
}


// This test ensures that the fp32 division of a variable x by sqrt(a)
// is transformed into x multiplied by the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_var_div_sqrt_f32(
// CHECK-SAME:  %[[X:.*]]: f32,
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation of rsqrt(a)
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Finally, x is multiplied by rsqrt(a)
// CHECK: arith.mulf %[[X]],
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_var_div_sqrt_f32(%arg0: f32, %arg1: f32) -> f32 {
  // %arg0 = x
  // %arg1 = a
  %1 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %arg0, %1 fastmath<fast> : f32
  return %2 : f32
}

// -----

// Generic scalar division (without sqrt denominator): apply runtime sign
// transfer so the divide is always rewritten regardless of sign provability.
// CHECK-LABEL: func.func @test_generic_div_f32_kept(
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_generic_div_f32_kept(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.divf %arg0, %arg1 fastmath<fast> : f32
  return %0 : f32
}

// Provably non-negative denominator: generic rewrite is still allowed.
// Here q = abs(x), so 1/|q| == 1/q and the fast inverse sequence is valid.
// CHECK-LABEL: func.func @test_non_negative_div_rewritten(
// CHECK: %[[ABS:.*]] = math.absf %[[Q:.*]] : f32
// CHECK: arith.bitcast
// CHECK: arith.shrui
// CHECK: arith.subi %c1597463007_i32,
// CHECK: arith.mulf %[[X:.*]],
// CHECK-NOT: arith.divf
func.func @test_non_negative_div_rewritten(%arg0: f32, %arg1: f32) -> f32 {
  %0 = math.absf %arg1 : f32
  %1 = arith.divf %arg0, %0 : f32
  return %1 : f32
}

// -----

// -0.0 denominator: inferSign returns Unknown (isNegative() == true so it
// does not qualify as Positive). With runtime sign transfer the rewrite fires:
// rsqrt(-0.0 * -0.0) = rsqrt(0.0) constant-folds to NaN (Quake algorithm
// diverges at 0), so the entire sequence folds to `arg0 * NaN`. This is a
// known limitation for the degenerate 0 / -0.0 case.
// CHECK-LABEL: func.func @test_negative_zero_div_kept(
// CHECK:       arith.mulf
// CHECK-NOT:   arith.divf
func.func @test_negative_zero_div_kept(%arg0: f32) -> f32 {
  %nzero = arith.constant -0.000000e+00 : f32
  %0 = arith.divf %arg0, %nzero : f32
  return %0 : f32
}

// -----

// Broadcast path, provably non-negative denominator: the generic rewrite still
// fires. Mirror of @test_scalar_division_f32 but with denom = broadcast(abs(s)),
// so 1/|s| == 1/s and the fast inverse sequence is valid. The scalar denom is
// squared, fed through the Quake rsqrt approximation, broadcast, then multiplied
// by the numerator -- and no divf remains.
// CHECK-LABEL: func.func @test_scalar_division_f32_nonneg(
// CHECK-SAME:  %[[VAL_0:.*]]: vector<128xf32>, %[[VAL_1:.*]]: memref<f32>, %[[VAL_2:.*]]: memref<128xf32>)
// CHECK:       %[[S:.*]] = memref.load %[[VAL_1]][] : memref<f32>
// CHECK:       %[[Q:.*]] = math.absf %[[S]] : f32
// CHECK:       %[[Q2:.*]] = arith.mulf %[[Q]], %[[Q]] : f32
// CHECK:       arith.bitcast
// CHECK:       arith.shrui
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       %[[INV:.*]] = vector.broadcast %{{.*}} : f32 to vector<128xf32>
// CHECK:       %[[MUL:.*]] = arith.mulf %[[VAL_0]], %[[INV]] : vector<128xf32>
// CHECK:       vector.transfer_write %[[MUL]], %[[VAL_2]]
// CHECK-NOT:   arith.divf
func.func @test_scalar_division_f32_nonneg(%arg0: vector<128xf32>, %arg1: memref<f32>, %arg2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f32>
  // |scalar| >= 0, so the generic reciprocal rewrite is sign-correct.
  %nonneg = math.absf %scalar : f32
  %denominator = vector.broadcast %nonneg : f32 to vector<128xf32>
  %div = arith.divf %arg0, %denominator : vector<128xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf32>, memref<128xf32>
  return
}

// -----

// Generic vector path, provably non-negative denominator: the rewrite still
// fires. Mirror of @test_fast_inverse_division_f32 but with denom = abs(%arg1),
// so 1/|q| == 1/q elementwise and the fast inverse sequence is valid. The
// denominator vector is squared, fed through the Quake rsqrt approximation, and
// multiplied by the numerator -- and no divf remains.
// CHECK-LABEL:   func.func @test_fast_inverse_division_f32_nonneg(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<256xf32>, %[[VAL_1:.*]]: vector<256xf32>, %[[VAL_2:.*]]: memref<256xf32>)
// CHECK:         %[[Q:.*]] = math.absf %[[VAL_1]] : vector<256xf32>
// CHECK:         %[[Q2:.*]] = arith.mulf %[[Q]], %[[Q]] : vector<256xf32>
// CHECK:         arith.bitcast
// CHECK:         arith.shrui
// CHECK:         arith.subi
// CHECK:         %[[MUL:.*]] = arith.mulf %[[VAL_0]],
// CHECK:         vector.transfer_write %[[MUL]], %[[VAL_2]]
// CHECK-NOT:     arith.divf
func.func @test_fast_inverse_division_f32_nonneg(%arg0: vector<256xf32>, %arg1: vector<256xf32>, %arg2: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  // |%arg1| >= 0 elementwise, so the generic reciprocal rewrite is sign-correct.
  %nonneg = math.absf %arg1 : vector<256xf32>
  %div = arith.divf %arg0, %nonneg : vector<256xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<256xf32>, memref<256xf32>
  return
}

// -----

// Softmax normalization: out = exp(x) / sum(exp(x)). The denominator is a
// reduction-add over exp(...) terms, hence provably >= 0, so the fast-inverse
// rewrite MUST still fire on this hot path. isKnownNonNegative proves it by
// recursing broadcast -> reduction<add> -> exp. No divf must remain.
// CHECK-LABEL: func.func @test_softmax_denom_rewritten(
// CHECK:       math.exp
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK-NOT:   arith.divf
func.func @test_softmax_denom_rewritten(%x: vector<8xf32>) -> vector<8xf32> {
  %e = math.exp %x : vector<8xf32>
  %sum = vector.reduction <add>, %e : vector<8xf32> into f32
  %bsum = vector.broadcast %sum : f32 to vector<8xf32>
  %out = arith.divf %e, %bsum : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// Divide by a reduction sum of UNKNOWN sign: with runtime sign transfer the
// rewrite fires regardless -- Quake rsqrt + bitcast/andi/ori sign fixup.
// CHECK-LABEL: func.func @test_div_by_unknown_sum_kept(
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_div_by_unknown_sum_kept(%x: vector<8xf32>) -> vector<8xf32> {
  %sum = vector.reduction <add>, %x : vector<8xf32> into f32
  %bsum = vector.broadcast %sum : f32 to vector<8xf32>
  %out = arith.divf %x, %bsum : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// Fast-exp-approx path: exp is replaced by an opaque polynomial, but the result
// op is tagged hexagon.nonneg by FastExpApproxPass. isKnownNonNegative honors
// the marker, so the sum(approx_exp) denominator is still proved >= 0 and the
// softmax rewrite fires. Without the marker (no-marker counterpart below) the
// same polynomial would not be rewritten.
// CHECK-LABEL: func.func @test_fast_exp_approx_nonneg_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_fast_exp_approx_nonneg_rewritten(%num: f32, %x: f32) -> f32 {
  // Simulate the output of FastExpApproxPass (a mulf with hexagon.nonneg).
  %e = arith.mulf %x, %x {hexagon.nonneg} : f32
  %out = arith.divf %num, %e : f32
  return %out : f32
}

// -----

// Without the hexagon.nonneg marker the structural mulf has unknown sign;
// runtime sign transfer rewrites it via Quake rsqrt + bitcast/andi/ori.
// CHECK-LABEL: func.func @test_unmarked_mulf_kept(
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_unmarked_mulf_kept(%num: f32, %a: f32, %b: f32) -> f32 {
  // Two distinct arguments -- not a self-multiply, no hexagon.nonneg: unknown sign.
  %e = arith.mulf %a, %b : f32
  %out = arith.divf %num, %e : f32
  return %out : f32
}

// -----

// Self-multiply x*x is always >= 0: the rewrite must fire.
// CHECK-LABEL: func.func @test_self_multiply_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_self_multiply_rewritten(%num: f32, %x: f32) -> f32 {
  %sq = arith.mulf %x, %x : f32
  %out = arith.divf %num, %sq : f32
  return %out : f32
}

// -----

// math.exp2(x) is always > 0: the rewrite must fire.
// CHECK-LABEL: func.func @test_exp2_denom_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_exp2_denom_rewritten(%num: f32, %x: f32) -> f32 {
  %e = math.exp2 %x : f32
  %out = arith.divf %num, %e : f32
  return %out : f32
}

// -----

// addf of two non-negative (exp) operands is non-negative: the predicate
// recurses and the rewrite fires. Tests the recursive addf case exercised
// when individual exp terms are summed before a reduction.
// CHECK-LABEL: func.func @test_addf_of_exp_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_addf_of_exp_rewritten(%num: f32, %x: f32, %y: f32) -> f32 {
  %ex = math.exp %x : f32
  %ey = math.exp %y : f32
  %sum = arith.addf %ex, %ey : f32
  %out = arith.divf %num, %sum : f32
  return %out : f32
}

// -----

// vector.reduction <mul> over a non-negative vector is non-negative: the
// rewrite must fire (product of positive exp terms > 0).
// CHECK-LABEL: func.func @test_reduction_mul_exp_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_reduction_mul_exp_rewritten(%num: f32, %x: vector<8xf32>) -> f32 {
  %e = math.exp %x : vector<8xf32>
  %prod = vector.reduction <mul>, %e : vector<8xf32> into f32
  %out = arith.divf %num, %prod : f32
  return %out : f32
}

// -----

// Provably-negative denominator: negf(absf(x)) is always <= 0.
// p / q  with q < 0  -->  -(p * Quake_rsqrt(q*q))
// The rewrite must fire and insert an arith.negf on the product.
// CHECK-LABEL: func.func @test_neg_abs_denom_rewritten(
// CHECK:       %[[Q:.*]] = math.absf
// CHECK:       %[[NEGQ:.*]] = arith.negf %[[Q]] : f32
// CHECK:       %[[Q2:.*]] = arith.mulf %[[NEGQ]], %[[NEGQ]] : f32
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK:       arith.negf
// CHECK-NOT:   arith.divf
func.func @test_neg_abs_denom_rewritten(%num: f32, %x: f32) -> f32 {
  %a = math.absf %x : f32
  %q = arith.negf %a : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}

// -----

// Provably-negative denominator: negf(exp(x)) is always < 0.
// CHECK-LABEL: func.func @test_neg_exp_denom_rewritten(
// CHECK:       math.exp
// CHECK:       arith.negf
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK:       arith.negf
// CHECK-NOT:   arith.divf
func.func @test_neg_exp_denom_rewritten(%num: f32, %x: f32) -> f32 {
  %e = math.exp %x : f32
  %q = arith.negf %e : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}

// -----

// negf(negf(exp(x))) is provably positive: the arith canonicalizer folds
// negf(negf(x)) to x before FastInverse runs, so the denom reduces to exp(x)
// which inferSign proves Positive. The rewrite fires WITHOUT arith.negf.
// CHECK-LABEL: func.func @test_double_neg_exp_rewritten(
// CHECK:       math.exp
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK-NOT:   arith.negf
// CHECK-NOT:   arith.divf
func.func @test_double_neg_exp_rewritten(%num: f32, %x: f32) -> f32 {
  %e = math.exp %x : f32
  %ne = arith.negf %e : f32
  %nne = arith.negf %ne : f32
  %out = arith.divf %num, %nne : f32
  return %out : f32
}

// -----

// Broadcast of negf(exp(x)) is provably negative: the vectorized case inserts
// a negf on the scalar inverse BEFORE broadcast (not on the product vector).
// The Quake sequence runs on the scalar denom, the sign correction negates the
// scalar rsqrt result, then it is broadcast and multiplied.
// CHECK-LABEL: func.func @test_neg_exp_broadcast_rewritten(
// CHECK:       math.exp
// CHECK:       arith.negf
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.negf
// CHECK:       vector.broadcast
// CHECK:       %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>
// CHECK-NOT:   arith.negf
// CHECK-NOT:   arith.divf
func.func @test_neg_exp_broadcast_rewritten(%num: vector<8xf32>, %x: f32) -> vector<8xf32> {
  %e = math.exp %x : f32
  %ne = arith.negf %e : f32
  %bcast = vector.broadcast %ne : f32 to vector<8xf32>
  %out = arith.divf %num, %bcast : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// mulf(negf(exp(x)), exp(y)) is Neg * Pos = Neg: inferSign returns Negative
// and the rewrite inserts arith.negf on the product.
// CHECK-LABEL: func.func @test_neg_pos_mulf_denom_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK:       arith.negf
// CHECK-NOT:   arith.divf
func.func @test_neg_pos_mulf_denom_rewritten(%num: f32, %x: f32, %y: f32) -> f32 {
  %ex = math.exp %x : f32
  %nex = arith.negf %ex : f32
  %ey = math.exp %y : f32
  %q = arith.mulf %nex, %ey : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}

// -----

// mulf(negf(exp(x)), negf(exp(y))) is Neg * Neg = Pos: inferSign returns
// Positive and the rewrite fires WITHOUT a surrounding arith.negf.
// CHECK-LABEL: func.func @test_neg_neg_mulf_denom_positive(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       %[[RESULT:.*]] = arith.mulf
// CHECK-NOT:   arith.negf {{.*}}%[[RESULT]]
// CHECK-NOT:   arith.divf
func.func @test_neg_neg_mulf_denom_positive(%num: f32, %x: f32, %y: f32) -> f32 {
  %ex = math.exp %x : f32
  %ey = math.exp %y : f32
  %nex = arith.negf %ex : f32
  %ney = arith.negf %ey : f32
  %q = arith.mulf %nex, %ney : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}

// -----

// addf(negf(exp(x)), negf(exp(y))) is Neg + Neg = Neg: the addf negative
// branch of inferSign returns Negative and the rewrite fires with negf.
// CHECK-LABEL: func.func @test_addf_neg_neg_denom_rewritten(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.mulf
// CHECK:       arith.negf
// CHECK-NOT:   arith.divf
func.func @test_addf_neg_neg_denom_rewritten(%num: f32, %x: f32, %y: f32) -> f32 {
  %ex = math.exp %x : f32
  %ey = math.exp %y : f32
  %nex = arith.negf %ex : f32
  %ney = arith.negf %ey : f32
  %q = arith.addf %nex, %ney : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}

// -----

// vector.reduction <add> over negf(exp(v)) is Neg: the sign correction negates
// the scalar rsqrt result BEFORE broadcast (not on the product vector).
// CHECK-LABEL: func.func @test_neg_exp_reduction_rewritten(
// CHECK:       math.exp
// CHECK:       arith.negf
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.negf
// CHECK:       vector.broadcast
// CHECK:       %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>
// CHECK-NOT:   arith.negf
// CHECK-NOT:   arith.divf
func.func @test_neg_exp_reduction_rewritten(%num: vector<8xf32>, %x: vector<8xf32>) -> vector<8xf32> {
  %e = math.exp %x : vector<8xf32>
  %ne = arith.negf %e : vector<8xf32>
  %sum = vector.reduction <add>, %ne : vector<8xf32> into f32
  %bsum = vector.broadcast %sum : f32 to vector<8xf32>
  %out = arith.divf %num, %bsum : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// vector.shuffle passthrough: when both shuffle sources share a sign, every
// selected lane has that sign. This is the shape the
// BroadcastedReductionToButterfly pass produces for a softmax denominator --
// the broadcast(reduction(exp)) is rewritten to a lane-wise butterfly of
// vector.shuffle + arith.addf. Both shuffle operands here trace to exp (> 0),
// so inferSign proves Positive and the reciprocal rewrite fires.
// CHECK-LABEL: func.func @test_shuffle_positive_rewritten(
// CHECK:       arith.constant dense<1597463007>
// CHECK:       arith.bitcast
// CHECK:       arith.subi
// CHECK-NOT:   arith.divf
func.func @test_shuffle_positive_rewritten(%x: vector<4xf32>) -> vector<4xf32> {
  %e = math.exp %x : vector<4xf32>
  %t0 = vector.shuffle %e, %e [2, 3, 0, 1] : vector<4xf32>, vector<4xf32>
  %b0 = arith.addf %e, %t0 : vector<4xf32>
  %t1 = vector.shuffle %b0, %b0 [1, 0, 3, 2] : vector<4xf32>, vector<4xf32>
  %denom = arith.addf %b0, %t1 : vector<4xf32>
  %out = arith.divf %e, %denom : vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// vector.shuffle of UNKNOWN-sign sources: with runtime sign transfer the
// rewrite fires via Quake rsqrt + bitcast/andi/ori sign fixup.
// CHECK-LABEL: func.func @test_shuffle_unknown_kept(
// CHECK:       arith.subi
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_shuffle_unknown_kept(%x: vector<4xf32>) -> vector<4xf32> {
  %t0 = vector.shuffle %x, %x [2, 3, 0, 1] : vector<4xf32>, vector<4xf32>
  %b0 = arith.addf %x, %t0 : vector<4xf32>
  %t1 = vector.shuffle %b0, %b0 [1, 0, 3, 2] : vector<4xf32>, vector<4xf32>
  %denom = arith.addf %b0, %t1 : vector<4xf32>
  %out = arith.divf %x, %denom : vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// vector.extract_strided_slice passthrough: the butterfly replicates the
// reduced value up to native HVX width and then contracts back with an
// extract_strided_slice. inferSign must walk through both the widening
// shuffles and the slice to reach the Positive exp source.
// CHECK-LABEL: func.func @test_extract_strided_slice_positive_rewritten(
// CHECK:       arith.constant dense<1597463007>
// CHECK:       arith.bitcast
// CHECK:       arith.subi
// CHECK-NOT:   arith.divf
func.func @test_extract_strided_slice_positive_rewritten(%x: vector<8xf32>) -> vector<8xf32> {
  %e = math.exp %x : vector<8xf32>
  %wide = vector.shuffle %e, %e
      [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
      : vector<8xf32>, vector<8xf32>
  %denom = vector.extract_strided_slice %wide
      {offsets = [0], sizes = [8], strides = [1]}
      : vector<16xf32> to vector<8xf32>
  %out = arith.divf %e, %denom : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// End-to-end butterfly shape: the marked fast-exp approximation (hexagon.nonneg
// on a vector mulf) feeds a softmax denominator that the
// BroadcastedReductionToButterfly pass has lowered to shuffle + addf. The
// marker proves the leaf Positive and the passthrough rules carry it through
// the whole butterfly, so the reciprocal rewrite fires. This is the production
// softmax path when enableFastExpApprox + enableBroadcastedReductionToButterfly
// are on.
// CHECK-LABEL: func.func @test_marked_exp_butterfly_rewritten(
// CHECK:       arith.constant dense<1597463007>
// CHECK:       arith.bitcast
// CHECK:       arith.subi
// CHECK-NOT:   arith.divf
func.func @test_marked_exp_butterfly_rewritten(%x: vector<4xf32>) -> vector<4xf32> {
  %e = arith.mulf %x, %x {hexagon.nonneg} : vector<4xf32>
  %t0 = vector.shuffle %e, %e [2, 3, 0, 1] : vector<4xf32>, vector<4xf32>
  %b0 = arith.addf %e, %t0 : vector<4xf32>
  %t1 = vector.shuffle %b0, %b0 [1, 0, 3, 2] : vector<4xf32>, vector<4xf32>
  %denom = arith.addf %b0, %t1 : vector<4xf32>
  %out = arith.divf %e, %denom : vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// inferSign BOUNDARY (structured scf.for form) -- NOT a production coverage
// hole. Real online-softmax / flash-attention kernels DO get the reciprocal
// optimization; see @test_loop_carried_sum_denom_rewritten_cfg below.
//
// This is the online-softmax / flash-attention denominator expressed with a
// STRUCTURED scf.for. l_i starts at 0.0 and is updated as
//   l_i = l_i + sum(exp(...))
// so it is mathematically >= 0 (a running sum of exp terms, all > 0).
//
// inferSign() does not prove it in THIS form: it reasons about CFG block
// arguments (the shape produced by SCFToControlFlow), not scf.for iter-args /
// results, so the scf.for result reads back Unknown. This costs nothing in
// production, because createFastInversePass() runs AFTER two SCFToControlFlow
// passes in the real LinalgToLLVM pipeline (see LinalgToLLVMPass.cpp: the
// SCFToControlFlow passes precede createFastInversePass). Every structured
// scf.for a real kernel writes has ALREADY been lowered to the CFG loop shape
// by the time this pass runs, and that CFG shape IS optimized -- the
// loop-carried non-negativity proof (an optimistic greatest-fixpoint over the
// loop-carried block argument) lives entirely on the CFG path and is exercised
// by @test_loop_carried_sum_denom_rewritten_cfg below. Teaching inferSign the
// scf.for shape would mean pulling the scf dialect into a post-SCF-lowering
// pass to prove a shape that can never reach it -- pure dead weight.
//
// For the END-TO-END production proof -- an author-written structured scf.for
// run through the real convert-scf-to-cf lowering and then fast-inverse, with
// the divide confirmed gone -- see fast_inverse_softmax_loop_pipeline.mlir.
//
// (If a producer ever hands this pass a structured scf.for directly, it can
// still opt in via the hexagon.nonneg attribute; see
// @test_loop_carried_marked_sum_rewritten below.)
//
// The scf.for iter-arg has Unknown sign (FastInverse reasons only about CFG
// block args). With runtime sign transfer the rewrite still fires: the
// scf.for survives but the divide is replaced with Quake rsqrt + andi/ori.
// CHECK-LABEL: func.func @test_loop_carried_sum_denom_kept_scf(
// CHECK:       scf.for
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_loop_carried_sum_denom_kept_scf(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e = math.exp %x : vector<8xf32>
  // l_i >= 0: init 0.0, each step adds a reduction<add> over exp(...) (all > 0).
  %l = scf.for %i = %c0 to %n step %c1 iter_args(%li = %zero) -> (f32) {
    %ex = math.exp %x : vector<8xf32>
    %lij = vector.reduction <add>, %ex : vector<8xf32> into f32
    %next = arith.addf %li, %lij : f32
    scf.yield %next : f32
  }
  %bl = vector.broadcast %l : f32 to vector<8xf32>
  %out = arith.divf %e, %bl : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// Loop-carried accumulator, post-SCFToControlFlow CFG form -- the shape
// FastInverse ACTUALLY sees in the real LinalgToLLVM pipeline
// (createFastInversePass() runs after two SCFToControlFlow passes, so the
// scf.for has been lowered to cf.br / cf.cond_br and the accumulator is an
// unstructured block argument %li in ^bb1).
//
// This is the same online-softmax / flash-attention denominator as the scf
// test above: l_i starts at 0.0 and each iteration adds sum(exp(...)) (all
// > 0), so l_i is provably >= 0. inferSign() proves it with an optimistic-init
// greatest-fixpoint over the CFG block arguments: %li is seeded Positive, then
// refined as the meet of the values its predecessors forward to it -- the
// entry init (0.0 -> Positive) and the loop backedge addf(%li, reduction<add>(
// exp(...))) which is addf(Positive, Positive) = Positive under the running
// estimate. The fixpoint holds, %li is proved >= 0, and the reciprocal rewrite
// fires. This is the perf win on the flash-attention / online-softmax hot path.
// CHECK-LABEL: func.func @test_loop_carried_sum_denom_rewritten_cfg(
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_loop_carried_sum_denom_rewritten_cfg(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e = math.exp %x : vector<8xf32>
  cf.br ^bb1(%c0, %zero : index, f32)
^bb1(%i: index, %li: f32):  // 2 preds: entry, ^bb2
  %cond = arith.cmpi slt, %i, %n : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %ex = math.exp %x : vector<8xf32>
  %lij = vector.reduction <add>, %ex : vector<8xf32> into f32
  %next = arith.addf %li, %lij : f32
  %inext = arith.addi %i, %c1 : index
  cf.br ^bb1(%inext, %next : index, f32)
^bb3:  // pred: ^bb1
  %bl = vector.broadcast %li : f32 to vector<8xf32>
  %out = arith.divf %e, %bl : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// NEGATIVE CONTROL for the CFG fixpoint: the same loop shape, but the
// accumulator is seeded from and updated with UNKNOWN-sign values (a running
// sum over the raw input %x, which may be negative). The optimistic seed meets
// down to Unknown. With runtime sign transfer the rewrite still fires via
// Quake rsqrt + bitcast/andi/ori sign fixup.
// CHECK-LABEL: func.func @test_loop_carried_unknown_denom_kept_cfg(
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_loop_carried_unknown_denom_kept_cfg(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  cf.br ^bb1(%c0, %zero : index, f32)
^bb1(%i: index, %li: f32):  // 2 preds: entry, ^bb2
  %cond = arith.cmpi slt, %i, %n : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  // Sum over the raw input (unknown sign), NOT over exp(...).
  %lij = vector.reduction <add>, %x : vector<8xf32> into f32
  %next = arith.addf %li, %lij : f32
  %inext = arith.addi %i, %c1 : index
  cf.br ^bb1(%inext, %next : index, f32)
^bb3:  // pred: ^bb1
  %bl = vector.broadcast %li : f32 to vector<8xf32>
  %out = arith.divf %x, %bl : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// CONTROL for the scf known-gap test above: the exact same structured loop,
// but with the accumulator's non-negativity advertised via hexagon.nonneg on
// the scf.for op itself. inferSign honors hexagon.nonneg on any defining op,
// and the divf denominator's defining op (after broadcast) is the scf.for, so
// the marker is seen and the rewrite fires through the loop even though the
// structural scf.for rule is intentionally absent.
// CHECK-LABEL: func.func @test_loop_carried_marked_sum_rewritten(
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @test_loop_carried_marked_sum_rewritten(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e = math.exp %x : vector<8xf32>
  %l = scf.for %i = %c0 to %n step %c1 iter_args(%li = %zero) -> (f32) {
    %ex = math.exp %x : vector<8xf32>
    %lij = vector.reduction <add>, %ex : vector<8xf32> into f32
    %next = arith.addf %li, %lij : f32
    scf.yield %next : f32
  } {hexagon.nonneg}
  %bl = vector.broadcast %l : f32 to vector<8xf32>
  %out = arith.divf %e, %bl : vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// ub.poison as V2 of vector.shuffle: the butterfly rewrite produces
// shuffle(exp_vec, ub.poison, mask) where only V1 lanes are selected.
// Treating poison as Positive prevents Unknown from poisoning the sign chain.
// CHECK-LABEL: func.func @test_positive_sign_through_poison_shuffle_divf_rewritten
// CHECK-NOT:   arith.divf
// CHECK:       arith.mulf
func.func @test_positive_sign_through_poison_shuffle_divf_rewritten(%x: vector<4xf32>) -> f32 {
  %exp = math.exp %x : vector<4xf32>
  %poison = ub.poison : vector<4xf32>
  %shuffled = vector.shuffle %exp, %poison [0, 1, 2, 3]
      : vector<4xf32>, vector<4xf32>
  %c0 = arith.constant 0.0 : f32
  %sum = vector.reduction <add>, %shuffled, %c0 : vector<4xf32> into f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %sum : f32
  return %result : f32
}

// -----

// Both V1 and V2 of the shuffle are unknown sign (function arguments): with
// runtime sign transfer the rewrite fires via Quake rsqrt + andi/ori fixup.
// CHECK-LABEL: func.func @test_unknown_sign_through_shuffle_divf_kept
// CHECK:       arith.subi
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_unknown_sign_through_shuffle_divf_kept(%v1: vector<4xf32>,
                                           %v2: vector<4xf32>) -> f32 {
  %shuffled = vector.shuffle %v1, %v2 [0, 1, 2, 3]
      : vector<4xf32>, vector<4xf32>
  %c0 = arith.constant 0.0 : f32
  %sum = vector.reduction <add>, %shuffled, %c0 : vector<4xf32> into f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %sum : f32
  return %result : f32
}

// -----

// arith.extf propagates sign: widening a positive f16 to f32 stays positive.
// CHECK-LABEL: func.func @test_positive_sign_through_extf_divf_rewritten
// CHECK-NOT:   arith.divf
// CHECK:       arith.mulf
func.func @test_positive_sign_through_extf_divf_rewritten(%x: f16) -> f32 {
  %exp_f16 = math.exp %x : f16
  %exp_f32 = arith.extf %exp_f16 : f16 to f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %exp_f32 : f32
  return %result : f32
}

// -----

// extf of an unknown-sign input stays Unknown: runtime sign transfer rewrites
// it via Quake rsqrt + bitcast/andi/ori sign fixup.
// CHECK-LABEL: func.func @test_unknown_sign_through_extf_divf_kept
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_unknown_sign_through_extf_divf_kept(%x: f16) -> f32 {
  %widened = arith.extf %x : f16 to f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %widened : f32
  return %result : f32
}

// -----

// vector.transfer_read from a local alloca: when all transfer_write ops into
// the alloca store non-negative values, the loaded value is also non-negative.
// This bridges the sign gap when SplitReduceGeneric + bufferization spill
// exp() results to scratch memory.
// CHECK-LABEL: func.func @test_positive_sign_through_alloca_divf_rewritten
// CHECK-NOT:   arith.divf
// CHECK:       arith.mulf
func.func @test_positive_sign_through_alloca_divf_rewritten(%x: vector<4xf32>) -> f32 {
  %alloca = memref.alloca() : memref<4xf32>
  %exp = math.exp %x : vector<4xf32>
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  vector.transfer_write %exp, %alloca[%c0] : vector<4xf32>, memref<4xf32>
  %loaded = vector.transfer_read %alloca[%c0], %pad : memref<4xf32>, vector<4xf32>
  %zero = arith.constant 0.0 : f32
  %sum = vector.reduction <add>, %loaded, %zero : vector<4xf32> into f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %sum : f32
  return %result : f32
}

// -----

// Value written to alloca has unknown sign: runtime sign transfer rewrites via
// Quake rsqrt + bitcast/andi/ori sign fixup.
// CHECK-LABEL: func.func @test_unknown_sign_through_alloca_divf_kept
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_unknown_sign_through_alloca_divf_kept(%x: vector<4xf32>) -> f32 {
  %alloca = memref.alloca() : memref<4xf32>
  %c0 = arith.constant 0 : index
  vector.transfer_write %x, %alloca[%c0] : vector<4xf32>, memref<4xf32>
  %pad = arith.constant 0.0 : f32
  %loaded = vector.transfer_read %alloca[%c0], %pad : memref<4xf32>, vector<4xf32>
  %zero = arith.constant 0.0 : f32
  %sum = vector.reduction <add>, %loaded, %zero : vector<4xf32> into f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %sum : f32
  return %result : f32
}

// -----

// No writer to alloca — sign is unknown even though alloca exists.
// Runtime sign transfer rewrites via Quake rsqrt + bitcast/andi/ori fixup.
// CHECK-LABEL: func.func @test_no_writer_alloca_divf_kept
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @test_no_writer_alloca_divf_kept() -> f32 {
  %alloca = memref.alloca() : memref<4xf32>
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %loaded = vector.transfer_read %alloca[%c0], %pad : memref<4xf32>, vector<4xf32>
  %zero = arith.constant 0.0 : f32
  %sum = vector.reduction <add>, %loaded, %zero : vector<4xf32> into f32
  %one = arith.constant 1.0 : f32
  %result = arith.divf %one, %sum : f32
  return %result : f32
}
