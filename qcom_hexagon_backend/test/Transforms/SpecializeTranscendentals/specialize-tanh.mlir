//===- specialize-tanh.mlir -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -specialize-transcendentals -canonicalize | FileCheck %s

// Positive tests: each function encodes a valid tanh expansion using named
// linalg ops.  After the pass + canonicalize the dead intermediate ops are
// eliminated and only a linalg.generic(math.tanh) over the original input
// survives.
//
// Recognized patterns:
//   Form A:  (1   - exp(-2x)) / (1   + exp(-2x))   fill(1) is sub op-0
//   Form B:  (exp(2x) - 1)   / (exp(2x) + 1)       fill(1) is sub op-1
//   Form C:  (exp(x) - exp(-x)) / (exp(x) + exp(-x))  negation via linalg.negf
//   Form D:  (exp(x) - exp(-x)) / (exp(x) + exp(-x))  negation via linalg.mul(-1)

// -----
// Form A: (1 - exp(-2x)) / (1 + exp(-2x))
// fill(1) is sub operand 0 → multiplier must be -2.

// CHECK-LABEL: func.func @tanh_form_a
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       linalg.generic {{.*}} ins(%arg0 : tensor<128xf32>)
// CHECK:         math.tanh
func.func @tanh_form_a(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %cst_neg2 = arith.constant -2.000000e+00 : f32
  %cst_1    = arith.constant  1.000000e+00 : f32
  %fill_neg2 = linalg.fill ins(%cst_neg2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1    = linalg.fill ins(%cst_1    : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul  = linalg.mul  ins(%x,      %fill_neg2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp  ins(%mul               : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub  = linalg.sub  ins(%fill_1, %exp      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add  = linalg.add  ins(%fill_1, %exp      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div  ins(%sub,    %add      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Form B: (exp(2x) - 1) / (exp(2x) + 1)
// fill(1) is sub operand 1 → multiplier must be +2.  Constant on the right.

// CHECK-LABEL: func.func @tanh_form_b_const_right
// CHECK-NOT:   arith.divf
// CHECK:       math.tanh
func.func @tanh_form_b_const_right(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %cst_2 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %fill_2 = linalg.fill ins(%cst_2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1 = linalg.fill ins(%cst_1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul = linalg.mul ins(%x,   %fill_2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp = linalg.exp ins(%mul          : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub = linalg.sub ins(%exp, %fill_1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add = linalg.add ins(%exp, %fill_1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div = linalg.div ins(%sub, %add    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Form B variant: constant on the left (2 * x).
// Exercises the isFromFillWithConstant(m0, requiredMul) branch.

// CHECK-LABEL: func.func @tanh_form_b_const_left
// CHECK-NOT:   arith.divf
// CHECK:       math.tanh
func.func @tanh_form_b_const_left(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %cst_2 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %fill_2 = linalg.fill ins(%cst_2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill_1 = linalg.fill ins(%cst_1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul = linalg.mul ins(%fill_2, %x      : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp = linalg.exp ins(%mul             : tensor<128xf32>)                   outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub = linalg.sub ins(%exp,    %fill_1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add = linalg.add ins(%exp,    %fill_1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div = linalg.div ins(%sub,    %add    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Form C: (exp(x) - exp(-x)) / (exp(x) + exp(-x)), negation via linalg.negf.

// CHECK-LABEL: func.func @tanh_dual_negf
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       linalg.generic {{.*}} ins(%arg0 : tensor<128xf32>)
// CHECK:         math.tanh
func.func @tanh_dual_negf(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty   = tensor.empty() : tensor<128xf32>
  %exp_x   = linalg.exp  ins(%x              : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %neg_x   = linalg.negf ins(%x              : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp_nx  = linalg.exp  ins(%neg_x          : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub     = linalg.sub  ins(%exp_x, %exp_nx : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add     = linalg.add  ins(%exp_x, %exp_nx : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div     = linalg.div  ins(%sub,   %add    : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Form D: (exp(x) - exp(-x)) / (exp(x) + exp(-x)), negation via linalg.mul(x, -1).

// CHECK-LABEL: func.func @tanh_dual_mul_neg1
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       linalg.generic {{.*}} ins(%arg0 : tensor<128xf32>)
// CHECK:         math.tanh
func.func @tanh_dual_mul_neg1(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty    = tensor.empty() : tensor<128xf32>
  %cst_neg1 = arith.constant -1.000000e+00 : f32
  %fill_neg1 = linalg.fill ins(%cst_neg1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp_x    = linalg.exp  ins(%x               : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %neg_x    = linalg.mul  ins(%x,    %fill_neg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp_nx   = linalg.exp  ins(%neg_x            : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %sub      = linalg.sub  ins(%exp_x, %exp_nx   : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %add      = linalg.add  ins(%exp_x, %exp_nx   : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div      = linalg.div  ins(%sub,   %add       : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}
