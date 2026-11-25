//===- fast_inverse_softmax_pipeline.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Integration test for the cross-pass hexagon.nonneg contract.
//
// FastExpApprox replaces math.exp with an opaque polynomial / IEEE-bit-trick
// and tags the result mulf with hexagon.nonneg so that the sign-sensitive
// FastInversePass can still prove an exp-derived denominator is >= 0 and keep
// the softmax reciprocal optimization. Many passes run between the two in the
// real LinalgToLLVM pipeline (canonicalize x many, cse, vectorization,
// bufferization, ...). The standalone fast-exp-approx.mlir / fast_inverse.mlir
// tests only check the two endpoints in isolation; this test pins the *transit*
// so a future pass that drops discardable attributes trips a visible failure.
//
// The marker is produced by the real fast-exp-approx pass (not hand-written),
// then carried through the repeated canonicalize + cse passes, and finally
// consumed by fast-inverse. The exp-derived denominator must be rewritten to
// the Quake rsqrt reciprocal (magic constant 1597463007) with no arith.divf
// left, proving the marker both survived and was detected.

// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(fast-exp-approx),func.func(canonicalize,cse),hexagon-fast-inverse)' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @exp_denom_marker_survives
// CHECK-NOT:   math.exp
// CHECK:       arith.bitcast
// CHECK:       arith.subi %c1597463007_i32,
// CHECK-NOT:   arith.divf
func.func @exp_denom_marker_survives(%num: f32, %x: f32) -> f32 {
  %e = math.exp %x : f32
  %out = arith.divf %num, %e : f32
  return %out : f32
}

// -----

// Negative control: a denominator with no exp and no marker has unknown sign.
// With runtime sign transfer, the divide is now rewritten via Quake rsqrt +
// bitcast/andi/ori sign fixup rather than preserved as arith.divf.

// CHECK-LABEL: func.func @unmarked_denom_kept
// CHECK:       arith.subi %c1597463007_i32,
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @unmarked_denom_kept(%num: f32, %a: f32, %b: f32) -> f32 {
  %q = arith.mulf %a, %b : f32
  %out = arith.divf %num, %q : f32
  return %out : f32
}
