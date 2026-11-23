//===- fast-exp-approx.mlir -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -fast-exp-approx | FileCheck %s

// Smoke test that the FastExpApprox pass rewrites scalar f32 math.exp /
// math.exp2 inside linalg.generic bodies into polynomial + IEEE-bit-pack
// arithmetic. The @exp_f32 test pins the full default-degree-3 polynomial
// (1/ln2 multiplier + Horner coefficients on [-1, 0]) so any unintended
// change to the coefficients trips a test failure. The remaining tests just
// check that math.exp / math.exp2 are gone and that the canonical
// instructions of the rewrite (fptosi, sitofp, shli 23, bitcast i32 -> f32)
// are present.

// CHECK-LABEL: func.func @exp_f32
// CHECK-NOT:   math.exp
// CHECK:       linalg.generic
// CHECK:         %[[INVLN2:.*]] = arith.constant 1.44269502 : f32
// CHECK:         %[[Y:.*]] = arith.mulf %{{.*}}, %[[INVLN2]] : f32
// CHECK:         %[[KI:.*]] = arith.fptosi %[[Y]] : f32 to i32
// CHECK:         %[[KF:.*]] = arith.sitofp %[[KI]] : i32 to f32
// CHECK:         %[[F:.*]] = arith.subf %[[Y]], %[[KF]] : f32
// CHECK:         %[[C3:.*]] = arith.constant 0.0401568189 : f32
// CHECK:         %[[T0:.*]] = arith.mulf %[[C3]], %[[F]] : f32
// CHECK:         %[[C2:.*]] = arith.constant 0.23192814 : f32
// CHECK:         %[[T1:.*]] = arith.addf %[[T0]], %[[C2]] : f32
// CHECK:         %[[T2:.*]] = arith.mulf %[[T1]], %[[F]] : f32
// CHECK:         %[[C1:.*]] = arith.constant 0.691834331 : f32
// CHECK:         %[[T3:.*]] = arith.addf %[[T2]], %[[C1]] : f32
// CHECK:         %[[T4:.*]] = arith.mulf %[[T3]], %[[F]] : f32
// CHECK:         %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[TWOPOWF:.*]] = arith.addf %[[T4]], %[[ONE]] : f32
// CHECK:         %[[C127:.*]] = arith.constant 127 : i32
// CHECK:         %[[KBIAS:.*]] = arith.addi %[[KI]], %[[C127]] : i32
// CHECK:         %[[C23:.*]] = arith.constant 23 : i32
// CHECK:         %[[KSHIFT:.*]] = arith.shli %[[KBIAS]], %[[C23]] : i32
// CHECK:         %[[TWOPOWK:.*]] = arith.bitcast %[[KSHIFT]] : i32 to f32
// CHECK:         arith.mulf %[[TWOPOWF]], %[[TWOPOWK]] {hexagon.nonneg} : f32
func.func @exp_f32(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %r = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%x : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<128xf32>
  return %r : tensor<128xf32>
}

// -----

// CHECK-LABEL: func.func @exp2_f32
// CHECK-NOT:   math.exp2
// CHECK:       linalg.generic
// CHECK:         arith.fptosi
// CHECK:         arith.sitofp
// CHECK:         arith.shli {{.*}}, %{{.*}} : i32
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
func.func @exp2_f32(%x: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() : tensor<128xf32>
  %r = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%x : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp2 %in : f32
    linalg.yield %e : f32
  } -> tensor<128xf32>
  return %r : tensor<128xf32>
}

// -----

// fp16 inputs are intentionally not rewritten (the bit-trick assumes the
// f32 exponent layout). The pass should leave math.exp on f16 alone.

// CHECK-LABEL: func.func @exp_f16_not_rewritten
// CHECK:       math.exp
// CHECK-NOT:   arith.bitcast {{.*}} : i32 to f32
func.func @exp_f16_not_rewritten(%x: tensor<128xf16>) -> tensor<128xf16> {
  %empty = tensor.empty() : tensor<128xf16>
  %r = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%x : tensor<128xf16>) outs(%empty : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %e = math.exp %in : f16
    linalg.yield %e : f16
  } -> tensor<128xf16>
  return %r : tensor<128xf16>
}

// -----

// The result of the approximation must carry hexagon.nonneg so that
// sign-sensitive downstream rewrites (FastInversePass) can still treat an
// exp approximation as non-negative. Check that the attribute is present on
// the final mulf (twoPowF * twoPowK) for both math.exp and math.exp2.

// CHECK-LABEL: func.func @exp_nonneg_attr
// CHECK-NOT:   math.exp
// CHECK:       arith.mulf {{.*}} {hexagon.nonneg}
func.func @exp_nonneg_attr(%x: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %r = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%x : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @exp2_nonneg_attr
// CHECK-NOT:   math.exp2
// CHECK:       arith.mulf {{.*}} {hexagon.nonneg}
func.func @exp2_nonneg_attr(%x: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %r = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%x : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp2 %in : f32
    linalg.yield %e : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}
