//===- hoist_loop_invariant_alloc.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-hoist-loop-invariant-alloc -canonicalize | FileCheck %s

// Tests for HoistLoopInvariantAllocPattern.  A static-shaped memref.alloc
// inside an scf.for loop (directly or nested in an scf.if) whose result
// does not escape its parent block is hoisted to immediately before the
// scf.for so it allocates once instead of every iteration.

// -----

// Test 1: alloc directly inside an scf.for is hoisted before the for.
func.func @hoist_alloc_directly_in_for(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  scf.for %i = %lb to %ub step %step {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
  }
  return
}

// CHECK-LABEL: @hoist_alloc_directly_in_for
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: scf.for
// CHECK:   vector.transfer_write %{{.*}}, %[[ALLOC]]
// CHECK: return

// -----

// Test 2: alloc inside scf.if inside scf.for is hoisted before the for.
func.func @hoist_alloc_in_if_in_for(%cond: i1, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<2.000000e+00> : vector<32xf32>
  scf.for %i = %lb to %ub step %step {
    scf.if %cond {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
    }
  }
  return
}

// CHECK-LABEL: @hoist_alloc_in_if_in_for
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: scf.for
// CHECK:   scf.if
// CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]
// CHECK: return

// -----

// Test 3: Negative — alloc is yielded out of an scf.if inside the for, so
// it escapes its parent block.  Hoisting would change the semantics
// (every iteration would yield the same buffer instead of fresh ones).
func.func @negative_alloc_yielded(%cond: i1, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<3.000000e+00> : vector<32xf32>
  scf.for %i = %lb to %ub step %step {
    %res = scf.if %cond -> (memref<32xf32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
      scf.yield %alloc : memref<32xf32>
    } else {
      %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      scf.yield %alloc2 : memref<32xf32>
    }
    memref.dealloc %res : memref<32xf32>
  }
  return
}

// CHECK-LABEL: @negative_alloc_yielded
// CHECK: scf.for
// CHECK:   scf.if
// CHECK:     %[[A:.*]] = memref.alloc()
// CHECK:     vector.transfer_write %{{.*}}, %[[A]]
// CHECK:     scf.yield %[[A]]
// CHECK:   else
// CHECK:     %[[B:.*]] = memref.alloc()
// CHECK:     scf.yield %[[B]]
// CHECK:   memref.dealloc
// CHECK: return

// -----

// Test 4: Negative — alloc has a dynamic shape, so the size depends on a
// loop value and cannot be hoisted.  The alloc is copied to %out so the
// canonicalizer cannot DCE the body.
func.func @negative_dynamic_shape(%out: memref<?xf32>, %lb: index, %ub: index, %step: index, %sz: index) {
  %cst = arith.constant 4.000000e+00 : f32
  scf.for %i = %lb to %ub step %step {
    %alloc = memref.alloc(%sz) {alignment = 64 : i64} : memref<?xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<?xf32>)
    memref.copy %alloc, %out : memref<?xf32> to memref<?xf32>
  }
  return
}

// CHECK-LABEL: @negative_dynamic_shape
// CHECK: scf.for
// CHECK:   %[[ALLOC:.*]] = memref.alloc(%{{.*}})
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]]
// CHECK:   memref.copy %[[ALLOC]]
// CHECK: return

// -----

// Test 5: Positive — the buffer is consumed only through a memref.subview
// alias that stays inside the loop.  BufferViewFlowAnalysis follows the view,
// sees no escape, and the alloc is still hoisted.  Guards against an
// alias-aware check that over-conservatively refuses to hoist any aliased
// buffer.
func.func @hoist_alloc_used_via_subview(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<5.000000e+00> : vector<16xf32>
  scf.for %i = %lb to %ub step %step {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    %sv = memref.subview %alloc[0] [16] [1] : memref<32xf32> to memref<16xf32, strided<[1]>>
    vector.transfer_write %cst, %sv[%c0] : vector<16xf32>, memref<16xf32, strided<[1]>>
  }
  return
}

// CHECK-LABEL: @hoist_alloc_used_via_subview
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: scf.for
// CHECK:   %[[SV:.*]] = memref.subview %[[ALLOC]]
// CHECK:   vector.transfer_write %{{.*}}, %[[SV]]
// CHECK: return

// -----

// Test 6: Negative — a memref.cast alias of the alloc is yielded out of the
// scf.for and returned, so the buffer escapes the iteration.  The old
// lexical check only inspected direct users of the alloc (the cast, which is
// nested) and would have wrongly hoisted it; the alias-aware check follows
// the cast to the loop-carried yield and refuses.
func.func @negative_view_escapes_loop(%lb: index, %ub: index, %step: index, %init: memref<?xf32>) -> memref<?xf32> {
  %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (memref<?xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    %cast = memref.cast %alloc : memref<32xf32> to memref<?xf32>
    scf.yield %cast : memref<?xf32>
  }
  return %res : memref<?xf32>
}

// CHECK-LABEL: @negative_view_escapes_loop
// CHECK: scf.for
// CHECK:   %[[ALLOC:.*]] = memref.alloc()
// CHECK:   memref.cast %[[ALLOC]]
// CHECK: return

// -----

// Test 7: A non-escaping stack allocation is also safe to hoist. This is
// required for vector temporaries materialized inside loop bodies.
func.func @hoist_alloca_in_for(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<6.000000e+00> : vector<32xf32>
  scf.for %i = %lb to %ub step %step {
    %alloca = memref.alloca() : memref<32xf32>
    vector.transfer_write %cst, %alloca[%c0] : vector<32xf32>, memref<32xf32>
  }
  return
}

// CHECK-LABEL: @hoist_alloca_in_for
// CHECK: %[[ALLOCA:.*]] = memref.alloca()
// CHECK: scf.for
// CHECK:   vector.transfer_write %{{.*}}, %[[ALLOCA]]
// CHECK: return

// -----

// Test 8: Negative — an explicit dealloc inside the loop is an ownership
// boundary. Hoisting the alloc while leaving the dealloc in the loop would
// free the same allocation on every iteration.
func.func @negative_alloc_with_in_loop_dealloc(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<7.000000e+00> : vector<32xf32>
  scf.for %i = %lb to %ub step %step {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
    memref.dealloc %alloc : memref<32xf32>
  }
  return
}

// CHECK-LABEL: @negative_alloc_with_in_loop_dealloc
// CHECK: scf.for
// CHECK:   %[[ALLOC_DEALLOC:.*]] = memref.alloc()
// CHECK:   vector.transfer_write %{{.*}}, %[[ALLOC_DEALLOC]]
// CHECK:   memref.dealloc %[[ALLOC_DEALLOC]]
