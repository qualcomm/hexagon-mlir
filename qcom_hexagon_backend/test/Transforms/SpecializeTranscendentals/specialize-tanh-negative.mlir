//===- specialize-tanh-negative.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -specialize-transcendentals | FileCheck %s

// Negative tests: each function contains a div(sub, add) structure that
// resembles tanh but must NOT be matched.
//
// Two cases are direct regressions for the sign-mismatch bug: the old code
// accepted ±2 without checking the sign against the sub operand order, so
// it would silently rewrite -tanh(x) as tanh(x).

// CHECK-NOT: math.tanh

// -----
// Bug regression: (1 - exp(+2x)) / (1 + exp(+2x)) = -tanh(x), NOT tanh(x).
// fill(1) is sub operand 0 → valid only with exp(-2x); here the mul is +2.

// CHECK-LABEL: func.func @no_match_one_minus_exp_pos2x
// CHECK: linalg.div
func.func @no_match_one_minus_exp_pos2x(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty    = tensor.empty() : tensor<128xf32>
  %cst_pos2 = arith.constant  2.000000e+00 : f32  // wrong sign: should be -2
  %cst_1    = arith.constant  1.000000e+00 : f32
  %fill_2   = linalg.fill ins(%cst_pos2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1   = linalg.fill ins(%cst_1    : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul  = linalg.mul ins(%x,      %fill_2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%mul             : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub  = linalg.sub ins(%fill_1, %exp    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add  = linalg.add ins(%fill_1, %exp    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%sub,    %add    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Bug regression: (exp(-2x) - 1) / (exp(-2x) + 1) = -tanh(x), NOT tanh(x).
// fill(1) is sub operand 1 → valid only with exp(+2x); here the mul is -2.

// CHECK-LABEL: func.func @no_match_exp_neg2x_minus_one
// CHECK: linalg.div
func.func @no_match_exp_neg2x_minus_one(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty    = tensor.empty() : tensor<128xf32>
  %cst_neg2 = arith.constant -2.000000e+00 : f32  // wrong sign: should be +2
  %cst_1    = arith.constant  1.000000e+00 : f32
  %fill_neg2 = linalg.fill ins(%cst_neg2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1    = linalg.fill ins(%cst_1    : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul  = linalg.mul ins(%x,   %fill_neg2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%mul             : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub  = linalg.sub ins(%exp, %fill_1    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add  = linalg.add ins(%exp, %fill_1    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%sub, %add       : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Near-miss: numerator is correct but denominator uses fill(2) instead of fill(1).
// The add must share the same fill(1) as the sub; a different constant breaks the match.

// CHECK-LABEL: func.func @no_match_wrong_denom_const
// CHECK: linalg.div
func.func @no_match_wrong_denom_const(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty    = tensor.empty() : tensor<128xf32>
  %cst_neg2 = arith.constant -2.000000e+00 : f32
  %cst_1    = arith.constant  1.000000e+00 : f32
  %cst_2    = arith.constant  2.000000e+00 : f32
  %fill_neg2 = linalg.fill ins(%cst_neg2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1    = linalg.fill ins(%cst_1    : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_2    = linalg.fill ins(%cst_2    : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul  = linalg.mul ins(%x,      %fill_neg2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%mul               : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub  = linalg.sub ins(%fill_1, %exp      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add  = linalg.add ins(%fill_2, %exp      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>  // fill(2) ← wrong
  %div  = linalg.div ins(%sub,    %add      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Near-miss: div(sub, add) with no exp — pure linear arithmetic.
// (a - b) / (a + b) shares the structural shape but has no exponentials.

// CHECK-LABEL: func.func @no_match_no_exp
// CHECK: linalg.div
func.func @no_match_no_exp(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %sub = linalg.sub ins(%a, %b : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add = linalg.add ins(%a, %b : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div = linalg.div ins(%sub, %add : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}
