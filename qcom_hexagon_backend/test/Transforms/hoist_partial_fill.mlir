//===- hoist_partial_fill.mlir --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-hoist-loop-invariant-alloc -canonicalize | FileCheck %s

// Tests for HoistPartialFillPattern.  When a `linalg.fill ins(%cst) outs(%alloc)`
// inside a loop is followed by a `memref.copy` that overwrites a static prefix
// `subview(%alloc)[0][M][1]` of the alloc, the suffix `subview(%alloc)[M][N-M][1]`
// is invariant.  The pattern hoists a one-shot fill of the suffix before the
// outermost enclosing scf.for and removes the per-iteration fill.

// -----

// Test 1: a single enclosing scf.for; the suffix-fill is hoisted before it.
func.func @hoist_suffix_fill_basic(%feat: memref<80xf32>, %lb: index, %ub: index, %step: index) {
  %f0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
    %sv_dst = memref.subview %alloc[0] [80] [1] : memref<128xf32> to memref<80xf32, strided<[1]>>
    memref.copy %feat, %sv_dst : memref<80xf32> to memref<80xf32, strided<[1]>>
  }
  return
}

// CHECK-LABEL: @hoist_suffix_fill_basic
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[SV_SUFFIX:.*]] = memref.subview %[[ALLOC]][80] [48] [1]
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[SV_SUFFIX]]
// CHECK: scf.for
// CHECK-NOT: linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]]
// CHECK:   memref.copy
// CHECK: return

// -----

// Test 2: nested scf.for loops; the suffix-fill is hoisted before the
// outermost loop, not the immediately enclosing one.
func.func @hoist_suffix_fill_nested_loops(%feat: memref<80xf32>, %lb: index, %ub: index, %step: index) {
  %f0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
      %sv_dst = memref.subview %alloc[0] [80] [1] : memref<128xf32> to memref<80xf32, strided<[1]>>
      memref.copy %feat, %sv_dst : memref<80xf32> to memref<80xf32, strided<[1]>>
    }
  }
  return
}

// CHECK-LABEL: @hoist_suffix_fill_nested_loops
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[SV_SUFFIX:.*]] = memref.subview %[[ALLOC]][80] [48] [1]
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[SV_SUFFIX]]
// CHECK: scf.for
// CHECK:   scf.for
// CHECK-NOT: linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]]
// CHECK:     memref.copy
// CHECK: return

// -----

// Test 3: Negative — an extra transfer_write to the suffix portion of the
// alloc inside the loop body invalidates the hoist (the suffix is no
// longer loop-invariant zero).
func.func @negative_other_write_to_suffix(%feat: memref<80xf32>, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<1.000000e+00> : vector<48xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
    %sv_dst = memref.subview %alloc[0] [80] [1] : memref<128xf32> to memref<80xf32, strided<[1]>>
    memref.copy %feat, %sv_dst : memref<80xf32> to memref<80xf32, strided<[1]>>
    %sv_suffix = memref.subview %alloc[80] [48] [1] : memref<128xf32> to memref<48xf32, strided<[1], offset: 80>>
    vector.transfer_write %cst, %sv_suffix[%c0] : vector<48xf32>, memref<48xf32, strided<[1], offset: 80>>
  }
  return
}

// CHECK-LABEL: @negative_other_write_to_suffix
// CHECK: scf.for
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<128xf32>)
// CHECK:   memref.copy
// CHECK:   vector.transfer_write
// CHECK: return

// -----

// Test 4: Negative — the copy's target is a non-prefix subview (offset != 0),
// so the alloc cannot be partitioned into a copy-overwritten prefix and a
// loop-invariant suffix.
func.func @negative_non_prefix_offset(%feat: memref<64xf32>, %lb: index, %ub: index, %step: index) {
  %f0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
    %sv_dst = memref.subview %alloc[16] [64] [1] : memref<128xf32> to memref<64xf32, strided<[1], offset: 16>>
    memref.copy %feat, %sv_dst : memref<64xf32> to memref<64xf32, strided<[1], offset: 16>>
  }
  return
}

// CHECK-LABEL: @negative_non_prefix_offset
// CHECK: scf.for
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<128xf32>)
// CHECK:   memref.copy
// CHECK: return

// -----

// Test 5: Negative — a memref.store into the suffix region inside the loop.
// The old check only recognised vector.transfer_write / memref.copy writers
// and would have wrongly hoisted; the memory-effects check detects the store.
func.func @negative_store_to_suffix(%feat: memref<80xf32>, %lb: index, %ub: index, %step: index) {
  %c100 = arith.constant 100 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %f1 = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
    %sv_dst = memref.subview %alloc[0] [80] [1] : memref<128xf32> to memref<80xf32, strided<[1]>>
    memref.copy %feat, %sv_dst : memref<80xf32> to memref<80xf32, strided<[1]>>
    memref.store %f1, %alloc[%c100] : memref<128xf32>
  }
  return
}

// CHECK-LABEL: @negative_store_to_suffix
// CHECK: scf.for
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<128xf32>)
// CHECK:   memref.copy
// CHECK:   memref.store
// CHECK: return

// -----

// Test 6: Negative — a write to the suffix nested inside an scf.if in the
// loop body.  hasEffect ignores recursive effects, so this exercises the
// explicit walk into nested regions.
func.func @negative_write_nested_in_if(%feat: memref<80xf32>, %cond: i1, %lb: index, %ub: index, %step: index) {
  %c100 = arith.constant 100 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %f1 = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<128xf32>)
    %sv_dst = memref.subview %alloc[0] [80] [1] : memref<128xf32> to memref<80xf32, strided<[1]>>
    memref.copy %feat, %sv_dst : memref<80xf32> to memref<80xf32, strided<[1]>>
    scf.if %cond {
      memref.store %f1, %alloc[%c100] : memref<128xf32>
    }
  }
  return
}

// CHECK-LABEL: @negative_write_nested_in_if
// CHECK: scf.for
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<128xf32>)
// CHECK:   memref.copy
// CHECK:   scf.if
// CHECK:     memref.store
// CHECK: return
