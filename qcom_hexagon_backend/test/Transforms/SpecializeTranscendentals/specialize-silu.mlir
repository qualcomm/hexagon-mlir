//===- specialize-silu.mlir -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -specialize-transcendentals -canonicalize | FileCheck %s

// Positive tests: silu and sigmoid arithmetic expansions sharing the
// denominator (1 + exp(-x)).  There is no math.sigmoid op, so the pass
// rewrites both via the identity sigmoid(x) = 0.5*(1 + tanh(x/2)) and
// silu(x) = x * sigmoid(x), reusing the tanh library route.  After the pass +
// canonicalize the exp/div chain is gone and a math.tanh remains.
//
// Recognized forms:
//   silu, negf:      x / (1 + exp(negf(x)))
//   silu, mul(-1):   x / (1 + exp(mul(x,-1)))
//   silu, sub(0,x):  x / (1 + exp(sub(0,x)))   <- the shape the front-end emits
//   silu, const-order + f16 cast
//   sigmoid:         1 / (1 + exp(negf(x)))

// -----
// silu with sub(0, x) negation -- this is the negation form the Triton
// front-end actually lowers `-x` to (fill(0) - x), so this is the case that
// matters for the real silu_kernel.

// CHECK-LABEL: func.func @silu_sub_zero
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       math.tanh
func.func @silu_sub_zero(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill0 = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %negx = linalg.sub ins(%fill0, %x : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%negx : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%x, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// silu with negf, constant on the add left.

// CHECK-LABEL: func.func @silu_negf
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       math.tanh
func.func @silu_negf(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %negx = linalg.negf ins(%x : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp  ins(%negx : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add  ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div  ins(%x, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// silu with mul(x, -1) negation, constant on the add right.

// CHECK-LABEL: func.func @silu_mul_neg1
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       math.tanh
func.func @silu_mul_neg1(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %cneg1 = arith.constant -1.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fillneg1 = linalg.fill ins(%cneg1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %negx = linalg.mul ins(%x, %fillneg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%negx : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add ins(%exp, %fill1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%x, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// plain sigmoid: 1 / (1 + exp(-x)).  Numerator is fill(1), not x.

// CHECK-LABEL: func.func @sigmoid_negf
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       math.tanh
func.func @sigmoid_negf(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %negx = linalg.negf ins(%x : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp  ins(%negx : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add  ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div  ins(%fill1, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// f16 silu: exp computed in f32 then truncated back to f16 (the Triton
// `(-x).to(f32) ... .to(f16)` shape).  The pass must look through the casts
// and keep the tanh in f16 (native qhmath_hvx_tanh_ahf downstream).

// CHECK-LABEL: func.func @silu_f16_cast
// CHECK-NOT:   arith.divf
// CHECK-NOT:   math.exp
// CHECK:       math.tanh
func.func @silu_f16_cast(%x: tensor<128xf16>) -> tensor<128xf16> {
  %empty16 = tensor.empty() : tensor<128xf16>
  %empty32 = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f16
  %fill1 = linalg.fill ins(%c1 : f16) outs(%empty16 : tensor<128xf16>) -> tensor<128xf16>
  %negx = linalg.negf ins(%x : tensor<128xf16>) outs(%empty16 : tensor<128xf16>) -> tensor<128xf16>
  // cast f16 -> f32
  %negx32 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                            iterator_types = ["parallel"]}
      ins(%negx : tensor<128xf16>) outs(%empty32 : tensor<128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %e = arith.extf %in : f16 to f32
      linalg.yield %e : f32
  } -> tensor<128xf32>
  %exp32 = linalg.exp ins(%negx32 : tensor<128xf32>) outs(%empty32 : tensor<128xf32>) -> tensor<128xf32>
  // cast f32 -> f16
  %exp16 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
      ins(%exp32 : tensor<128xf32>) outs(%empty16 : tensor<128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %t = arith.truncf %in : f32 to f16
      linalg.yield %t : f16
  } -> tensor<128xf16>
  %den = linalg.add ins(%fill1, %exp16 : tensor<128xf16>, tensor<128xf16>) outs(%empty16 : tensor<128xf16>) -> tensor<128xf16>
  %div = linalg.div ins(%x, %den : tensor<128xf16>, tensor<128xf16>) outs(%empty16 : tensor<128xf16>) -> tensor<128xf16>
  return %div : tensor<128xf16>
}

// -----
// Negative test: numerator is an unrelated value (not x, not fill(1)), so the
// div is NOT silu/sigmoid and must be left as a division.

// CHECK-LABEL: func.func @not_silu_other_num
// CHECK:       linalg.div
// CHECK-NOT:   math.tanh
func.func @not_silu_other_num(%x: tensor<128xf32>, %y: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %negx = linalg.negf ins(%x : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp  ins(%negx : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add  ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div  ins(%y, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Negative test: denominator uses exp(-2x) (tanh-like, mul by -2), which is
// NOT a -1 negation, so the sigmoid-like matcher must reject it.

// CHECK-LABEL: func.func @not_silu_exp_2x
// CHECK:       linalg.div
// CHECK-NOT:   math.tanh
func.func @not_silu_exp_2x(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %cneg2 = arith.constant -2.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fillneg2 = linalg.fill ins(%cneg2 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %mul  = linalg.mul ins(%x, %fillneg2 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%mul : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%x, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}

// -----
// Negative test: exp argument is (x - 0), i.e. sub with the zero as the SECOND
// operand -- that is x, not -x, so this is NOT silu and must not rewrite.

// CHECK-LABEL: func.func @not_silu_sub_x_zero
// CHECK:       linalg.div
// CHECK-NOT:   math.tanh
func.func @not_silu_sub_x_zero(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %c1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0.000000e+00 : f32
  %fill1 = linalg.fill ins(%c1 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %fill0 = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %xm0  = linalg.sub ins(%x, %fill0 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %exp  = linalg.exp ins(%xm0 : tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %den  = linalg.add ins(%fill1, %exp : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %div  = linalg.div ins(%x, %den : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return %div : tensor<128xf32>
}
