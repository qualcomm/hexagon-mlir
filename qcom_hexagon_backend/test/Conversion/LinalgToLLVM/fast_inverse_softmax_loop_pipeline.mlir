//===- fast_inverse_softmax_loop_pipeline.mlir ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Production-shaped integration test for the loop-carried softmax / flash-
// attention reciprocal optimization.
//
// A kernel author writes the online-softmax denominator as a STRUCTURED
// scf.for: l_i starts at 0.0 and each iteration accumulates sum(exp(...)),
// so l_i is mathematically >= 0. In isolation, hexagon-fast-inverse does not
// reason about scf.for iter-args (see the @test_loop_carried_sum_denom_kept_scf
// boundary note in fast_inverse.mlir) -- but that never costs anything in
// production, because createFastInversePass() runs AFTER SCFToControlFlow in
// the real LinalgToLLVM pipeline (see LinalgToLLVMPass.cpp). By the time
// fast-inverse runs, the scf.for is already an unstructured CFG loop whose
// accumulator is a block argument, and inferSign()'s optimistic greatest-
// fixpoint over CFG block arguments DOES prove it >= 0.
//
// This test pins that exact production ordering end to end: it starts from the
// structured scf.for a kernel actually writes, runs the real convert-scf-to-cf
// lowering, then fast-inverse. The divide must be gone -- rewritten to the
// Quake rsqrt reciprocal (magic constant 1597463007) with no scf.for and no
// arith.divf left -- proving the flash-attention / online-softmax hot path
// really does get the optimization on the shape production emits.
//
// The standalone fast_inverse.mlir @test_loop_carried_sum_denom_rewritten_cfg
// test checks the post-lowering CFG shape directly; this test proves the SCF
// lowering actually produces that shape, so the two together close the loop
// from author-written source to optimized output.

// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(convert-scf-to-cf,hexagon-fast-inverse)' \
// RUN:   | FileCheck %s

// The scf.for must be lowered away by convert-scf-to-cf...
// CHECK-LABEL: func.func @softmax_loop_denom_rewritten
// CHECK-NOT:   scf.for
// ...and the loop-carried divide must be rewritten to the Quake rsqrt
// reciprocal, proving inferSign proved the CFG accumulator >= 0.
// CHECK:       %[[MAGIC:.*]] = arith.constant 1597463007 : i32
// CHECK:       arith.bitcast
// CHECK:       arith.subi %[[MAGIC]],
// CHECK:       vector.broadcast
// CHECK-NOT:   arith.divf
func.func @softmax_loop_denom_rewritten(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
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

// NEGATIVE CONTROL: identical structured loop and lowering, but the
// accumulator sums the RAW input %x (unknown sign) instead of exp(...). After
// convert-scf-to-cf the CFG fixpoint meets the accumulator block argument down
// to Unknown. With runtime sign transfer, the divide is now rewritten via
// Quake rsqrt + bitcast/andi/ori sign fixup rather than preserved as arith.divf.
// CHECK-LABEL: func.func @softmax_loop_unknown_denom_kept
// CHECK-NOT:   scf.for
// CHECK:       arith.subi {{.*}}1597463007{{.*}}
// CHECK:       arith.andi
// CHECK:       arith.ori
// CHECK-NOT:   arith.divf
func.func @softmax_loop_unknown_denom_kept(%x: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %l = scf.for %i = %c0 to %n step %c1 iter_args(%li = %zero) -> (f32) {
    // Sum over the raw input (unknown sign), NOT over exp(...).
    %lij = vector.reduction <add>, %x : vector<8xf32> into f32
    %next = arith.addf %li, %lij : f32
    scf.yield %next : f32
  }
  %bl = vector.broadcast %l : f32 to vector<8xf32>
  %out = arith.divf %x, %bl : vector<8xf32>
  return %out : vector<8xf32>
}
