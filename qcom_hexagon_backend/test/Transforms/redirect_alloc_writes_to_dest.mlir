//===- redirect_alloc_writes_to_dest.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Tests for the unified RedirectAllocWritesToDestinationPattern and the
// dedicated EliminateAllocInIfWithCopyPattern.  These patterns eliminate a
// trailing `memref.copy %alloc, %dst` when `%alloc` is a write-only staging
// buffer by redirecting in-scope writes to `%dst`.

// -----

// Test 1: alloc and copy in the same block, writes via subviews.
func.func @same_block_subview_writes(%dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %tile = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %cst, %sv[%c0] : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @same_block_subview_writes
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: scf.for %[[TILE:.*]] =
// CHECK:   %[[SV:.*]] = memref.subview %[[DST]][%[[TILE]]]
// CHECK:   vector.transfer_write %{{.*}}, %[[SV]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 2: alloc hoisted above the copy's enclosing loop, writes via subviews
// inside the copy's block.  An additional out-of-scope copy of the alloc keeps
// the alloc alive after the rewrite — only the in-scope inner copy is removed.
func.func @hoisted_alloc_subview_writes(%dst: memref<128xf32>, %ext_dst: memref<128xf32>, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  memref.copy %alloc, %ext_dst : memref<128xf32> to memref<128xf32>
  scf.for %pt = %lb to %ub step %step {
    scf.for %tile = %c0 to %c128 step %c32 {
      %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
      vector.transfer_write %cst, %sv[%c0] : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
    }
    memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  }
  return
}

// CHECK-LABEL: @hoisted_alloc_subview_writes
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32>, %[[EXT:.*]]: memref<128xf32>
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %[[ALLOC]], %[[EXT]]
// CHECK: scf.for
// CHECK:   scf.for %[[TILE:.*]] =
// CHECK:     %[[SV:.*]] = memref.subview %[[DST]][%[[TILE]]]
// CHECK:     vector.transfer_write %{{.*}}, %[[SV]]
// CHECK-NOT: memref.copy %[[ALLOC]], %[[DST]]
// CHECK: return

// -----

// Test 3: alloc hoisted above the copy's enclosing loop, written via a direct
// vector.transfer_write inside the copy's block.
func.func @hoisted_alloc_direct_transfer_write(%dst: memref<64xf32>, %ext_dst: memref<64xf32>, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<2.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %alloc, %ext_dst : memref<64xf32> to memref<64xf32>
  scf.for %i = %lb to %ub step %step {
    vector.transfer_write %cst, %alloc[%c0] : vector<64xf32>, memref<64xf32>
    memref.copy %alloc, %dst : memref<64xf32> to memref<64xf32>
  }
  return
}

// CHECK-LABEL: @hoisted_alloc_direct_transfer_write
// CHECK-SAME: (%[[DST:.*]]: memref<64xf32>, %[[EXT:.*]]: memref<64xf32>
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %[[ALLOC]], %[[EXT]]
// CHECK: scf.for
// CHECK:   vector.transfer_write %{{.*}}, %[[DST]]
// CHECK-NOT: memref.copy %[[ALLOC]], %[[DST]]
// CHECK: return

// -----

// Test 4: scf.if then-branch yields a fresh alloc filled by transfer_writes
// via subviews, else-branch yields the destination directly, and the if
// result is copied to the destination.  Covered by
// EliminateAllocInIfWithCopyPattern.
func.func @scf_if_yielded_alloc(%cond: i1, %dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<3.000000e+00> : vector<32xf32>
  %res = scf.if %cond -> (memref<128xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    scf.for %tile = %c0 to %c128 step %c32 {
      %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
      vector.transfer_write %cst, %sv[%c0] : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
    }
    scf.yield %alloc : memref<128xf32>
  } else {
    scf.yield %dst : memref<128xf32>
  }
  memref.copy %res, %dst : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @scf_if_yielded_alloc
// CHECK-SAME: (%{{.*}}: i1, %[[DST:.*]]: memref<128xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: scf.if
// CHECK:   scf.for %[[TILE:.*]] =
// CHECK:     %[[SV:.*]] = memref.subview %[[DST]][%[[TILE]]]
// CHECK:     vector.transfer_write %{{.*}}, %[[SV]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 5: Negative — a vector.transfer_read from the alloc disqualifies the
// rewrite.  Both the alloc and the trailing copy must remain.
func.func @negative_read_blocks_rewrite(%dst: memref<128xf32>) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<4.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  %read = vector.transfer_read %alloc[%c0], %f0 : memref<128xf32>, vector<128xf32>
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  return %read : vector<128xf32>
}

// CHECK-LABEL: @negative_read_blocks_rewrite
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[ALLOC]]
// CHECK: vector.transfer_read %[[ALLOC]]
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: return

// -----

// Test 6: Negative — alloc is hoisted above the copy's block but the only
// writes to it live OUTSIDE the copy's block.  Without any in-scope writes
// to redirect, the pattern leaves the alloc and the trailing copy alone.
func.func @negative_alloc_outside_no_writes_inside(%dst: memref<128xf32>, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<5.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  scf.for %i = %lb to %ub step %step {
    memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  }
  return
}

// CHECK-LABEL: @negative_alloc_outside_no_writes_inside
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[ALLOC]]
// CHECK: scf.for
// CHECK:   memref.copy %[[ALLOC]], %{{.*}}
// CHECK: return

// -----

// Test 7: Negative — dominance guard.  %dst is defined after %src's writer,
// so redirecting the writer to %dst would place a use before its definition;
// the pattern must bail.  The trailing write to %dst keeps CopyForwarding from
// folding the copy first, so the dominance guard is what's exercised.
func.func @negative_dest_after_writer(%later: vector<32xi32>) -> vector<32xi32> {
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %v = arith.constant dense<1> : vector<32xi32>
  %src = memref.alloc() {alignment = 64 : i64} : memref<32xi32, 1>
  vector.transfer_write %v, %src[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32, 1>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<32xi32, 1>
  memref.copy %src, %dst : memref<32xi32, 1> to memref<32xi32, 1>
  %read = vector.transfer_read %dst[%c0], %i0 {in_bounds = [true]} : memref<32xi32, 1>, vector<32xi32>
  vector.transfer_write %later, %dst[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32, 1>
  return %read : vector<32xi32>
}

// CHECK-LABEL: @negative_dest_after_writer
// CHECK: %[[SRC:.*]] = memref.alloc() {{.*}} : memref<32xi32, 1>
// CHECK: vector.transfer_write %{{.*}}, %[[SRC]]
// CHECK: %[[DST:.*]] = memref.alloc() {{.*}} : memref<32xi32, 1>
// CHECK: memref.copy %[[SRC]], %[[DST]]
// CHECK: vector.transfer_read %[[DST]]
// CHECK: return

// -----

// Test 8: alloc filled by `linalg.fill` (writes the alloc through
// MemoryEffectOpInterface without a subview or vector op).  The generalized
// pattern accepts any leaf op that declares Write-only effects on the operand,
// so the alloc + copy collapse into a fill that targets %dst directly.
func.func @same_block_linalg_fill(%dst: memref<64xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<64xf32>)
  memref.copy %alloc, %dst : memref<64xf32> to memref<64xf32>
  return
}

// CHECK-LABEL: @same_block_linalg_fill
// CHECK-SAME: (%[[DST:.*]]: memref<64xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: linalg.fill {{.*}} outs(%[[DST]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 9: alloc routed through `memref.expand_shape` (a non-subview
// ViewLikeOpInterface producer) before being written by a 2D
// vector.transfer_write.  The recursive view-chain check accepts the
// expand_shape and the alloc + copy collapse into a transfer_write whose
// expand_shape source is %dst.
func.func @same_block_expand_shape_chain(%dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<2.000000e+00> : vector<8x16xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  %expanded = memref.expand_shape %alloc [[0, 1]] output_shape [8, 16] : memref<128xf32> into memref<8x16xf32>
  vector.transfer_write %cst, %expanded[%c0, %c0] : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @same_block_expand_shape_chain
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: %[[EXP:.*]] = memref.expand_shape %[[DST]]
// CHECK: vector.transfer_write %{{.*}}, %[[EXP]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 10: multi-copy — the same alloc is drained into two destinations with
// disjoint writes between the two copies.  The pattern should run last-to-
// first under the greedy driver: copy2 redirects the second write to %dst2,
// then copy1 redirects the first write to %dst1, leaving no alloc and no
// copies behind.
func.func @same_block_multi_copy(%dst1: memref<128xf32>, %dst2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %cst1 = arith.constant dense<1.000000e+00> : vector<128xf32>
  %cst2 = arith.constant dense<2.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  vector.transfer_write %cst1, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  memref.copy %alloc, %dst1 : memref<128xf32> to memref<128xf32>
  vector.transfer_write %cst2, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  memref.copy %alloc, %dst2 : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @same_block_multi_copy
// CHECK-SAME: (%[[DST1:.*]]: memref<128xf32>, %[[DST2:.*]]: memref<128xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[DST1]]
// CHECK: vector.transfer_write %{{.*}}, %[[DST2]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 11: memory-space mismatch — alloc in the default space, dst in
// memory space 1.  Same shape, layout, and element type.  The pattern
// rebuilds the writer to target %dst directly in its native memory space
// (no memory_space_cast emitted).
func.func @same_block_memory_space_cast(%dst: memref<64xf32, 1>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<3.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<64xf32>, memref<64xf32>
  memref.copy %alloc, %dst : memref<64xf32> to memref<64xf32, 1>
  return
}

// CHECK-LABEL: @same_block_memory_space_cast
// CHECK-SAME: (%[[DST:.*]]: memref<64xf32, 1>)
// CHECK-NOT: memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[DST]]{{.*}} : vector<64xf32>, memref<64xf32, 1>
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 12 (negative): multi-copy where a `vector.transfer_read` of the alloc
// lies between the two copies.  Even though the read is outside the claimed
// range of either copy, the alloc is no longer write-only — redirecting any
// writer would change what the intervening read observes.  The pattern must
// bail and leave both copies and the alloc in place.
func.func @negative_multi_copy_with_read(%dst1: memref<32xf32>, %dst2: memref<32xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
  memref.copy %alloc, %dst1 : memref<32xf32> to memref<32xf32>
  %read = vector.transfer_read %alloc[%c0], %f0 : memref<32xf32>, vector<32xf32>
  memref.copy %alloc, %dst2 : memref<32xf32> to memref<32xf32>
  return %read : vector<32xf32>
}

// CHECK-LABEL: @negative_multi_copy_with_read
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[ALLOC]]
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: vector.transfer_read %[[ALLOC]]
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: return

// -----

// Test 13: memory-space mismatch combined with a subview chain.  DDR alloc,
// VTCM dst, writes go through `memref.subview`.  The subview should be
// rebuilt to take %dst as its source, with its result type's memory space
// tracking the new source — no memory_space_cast emitted.
func.func @same_block_subview_writes_to_vtcm(%dst: memref<128xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<0xFF800000> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %tile = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %cst, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32, 1>
  return
}

// CHECK-LABEL: @same_block_subview_writes_to_vtcm
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32, 1>)
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.memory_space_cast
// CHECK: scf.for %[[TILE:.*]] =
// CHECK:   %[[SV:.*]] = memref.subview %[[DST]][%[[TILE]]]{{.*}} : memref<128xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:   vector.transfer_write %{{.*}}, %[[SV]]{{.*}} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 14: memory-space mismatch combined with multi-drain shared staging.
// The same DDR alloc is reused for two sequential inits to two different
// VTCM destinations.  Each scf.for's subview is rebuilt against its drain's
// dest in VTCM.
func.func @same_block_shared_staging_to_vtcm(
    %dst_a: memref<128xf32, 1>,
    %dst_b: memref<128xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %neg = arith.constant dense<0xFF800000> : vector<32xf32>
  %zero = arith.constant dense<0.0> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>

  // First init: write -inf, copy to dst_a.
  scf.for %i = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%i] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %neg, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst_a : memref<128xf32> to memref<128xf32, 1>

  // Second init: overwrite with 0.0, copy to dst_b.
  scf.for %i = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%i] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %zero, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst_b : memref<128xf32> to memref<128xf32, 1>

  return
}

// CHECK-LABEL: @same_block_shared_staging_to_vtcm
// CHECK-SAME: (%[[DST_A:.*]]: memref<128xf32, 1>, %[[DST_B:.*]]: memref<128xf32, 1>)
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.memory_space_cast
// First init's writers should target dst_a.
// CHECK: scf.for
// CHECK:   memref.subview %[[DST_A]]{{.*}} : memref<128xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:   vector.transfer_write %{{.*}}{{.*}} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// Second init's writers should target dst_b.
// CHECK: scf.for
// CHECK:   memref.subview %[[DST_B]]{{.*}} : memref<128xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:   vector.transfer_write %{{.*}}{{.*}} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 15 (negative): coverage guard.  A single writer loop feeds the alloc,
// then two sibling drains read it.  Only the first drain can claim the writer,
// which would leave the second draining an alloc with no writers, so the
// pattern bails and leaves alloc + writer + both drains untouched.
func.func @negative_multi_drain_unclaimed_second(
    %dst1: memref<128xf32, 1>,
    %dst2: memref<128xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
  scf.for %tile = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    vector.transfer_write %cst, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
  }
  memref.copy %alloc, %dst1 : memref<128xf32, 1> to memref<128xf32, 1>
  memref.copy %alloc, %dst2 : memref<128xf32, 1> to memref<128xf32, 1>
  return
}

// CHECK-LABEL: @negative_multi_drain_unclaimed_second
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<128xf32, 1>
// CHECK: scf.for
// CHECK:   memref.subview %[[ALLOC]]
// CHECK:   vector.transfer_write
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: return

// -----

// Test 16 (negative): memory-hierarchy guard.  Alloc lives in VTCM (memory
// space 1), the trailing copy drains it to DDR (memory space 0).  This is
// the deliberate "stage hot in VTCM, then bulk copy to DDR" pattern.
// Redirecting the writers to DDR would lose the VTCM staging speed.  The
// pattern must bail.
func.func @negative_vtcm_to_ddr_preserves_staging(%dst: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<3.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, 1>
  memref.copy %alloc, %dst : memref<64xf32, 1> to memref<64xf32>
  return
}

// CHECK-LABEL: @negative_vtcm_to_ddr_preserves_staging
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<64xf32, 1>
// CHECK: vector.transfer_write %{{.*}}, %[[ALLOC]]{{.*}} : vector<64xf32>, memref<64xf32, 1>
// CHECK: memref.copy %[[ALLOC]], %{{.*}} : memref<64xf32, 1> to memref<64xf32>
// CHECK: return

// -----

// Test 17 (negative): cross-block multi-drain.  drain1 is inside the scf.for
// body (after the writer); drain2 follows the loop in the alloc's block and
// reads the last-iteration content.  Only drain1 can claim the writer, leaving
// drain2 unclaimed but originally reachable from it, so the pattern bails.
func.func @negative_cross_block_drains_bail(
    %dst1: memref<128xf32>, %dst2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %tile = %c0 to %c128 step %c32 {
    %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %cst, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
    memref.copy %alloc, %dst1 : memref<128xf32> to memref<128xf32>
  }
  memref.copy %alloc, %dst2 : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @negative_cross_block_drains_bail
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: scf.for
// CHECK:   memref.subview %[[ALLOC]]
// CHECK:   vector.transfer_write
// CHECK:   memref.copy %[[ALLOC]], %{{.*}}
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK: return

// -----

// Test 18: destination is a `memref.reinterpret_cast` output view with a
// DYNAMIC layout offset (e.g. program_id * BLOCK from a tiled gather store).
// The staging alloc differs from the dest only by that offset, so the redirect
// uses the dest directly and each cloned subview RE-INFERS its result type from
// the offset-carrying dest — preserving the runtime offset instead of casting
// it away.  (A `memref.cast` to the alloc's identity layout would drop the
// offset, making every tile clobber out[0..N-1]; that was the multi-tile gather
// miscompile this pattern used to introduce.)  The alloc and the trailing copy
// are eliminated and the writes target the dest slice at its real offset.
func.func @reinterpret_cast_dynamic_offset_dest(
    %out: memref<*xf32>, %pid: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<1.000000e+00> : vector<32xf32>
  %off = arith.muli %pid, %c128 : index
  %dst = memref.reinterpret_cast %out to offset: [%off], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  scf.for %tile = %c0 to %c128 step %c1 {
    %sv = memref.subview %alloc[%tile] [32] [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %cst, %sv[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32, strided<[1], offset: ?>>
  return
}

// The alloc and the trailing copy are gone; the writer subview is rebuilt on
// the offset-carrying dest (so the store lands at out[off ..]).
// CHECK-LABEL: @reinterpret_cast_dynamic_offset_dest
// CHECK: %[[DST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}]
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.cast
// CHECK: scf.for
// CHECK:   %[[SV:.*]] = memref.subview %[[DST]]
// CHECK:   vector.transfer_write %{{.*}}, %[[SV]]
// CHECK-NOT: memref.copy
// CHECK: return

// -----

// Test 19: writer is a RANK-REDUCING subview (2-D source, singleton leading
// dim dropped -> 1-D result), redirected onto a 2-D dest.  A plain
// (non-rank-reduced) re-inference of the writer's result type against the
// new 2-D source would keep it 2-D, leaving the `vector.transfer_write`
// (which supplies exactly 1 index, matching the ORIGINAL 1-D subview) with
// too few indices for its rebuilt memref operand -- a verifier failure
// ("'vector.transfer_write' op requires 2 indices") caught by hexagon-mlir
// CI on a real linear-attention kernel.  The redirect must reproduce the
// SAME rank (1-D) on the rebuilt subview, carrying dest's offset/strides.
func.func @rank_reducing_subview_writer(%dst: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
  scf.for %row = %c0 to %c64 step %c1 {
    %sv = memref.subview %alloc[%row, 0] [1, 64] [1, 1] : memref<64x64xf32> to memref<64xf32, strided<[1], offset: ?>>
    vector.transfer_write %cst, %sv[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1], offset: ?>>
  }
  memref.copy %alloc, %dst : memref<64x64xf32> to memref<64x64xf32>
  return
}

// The rebuilt subview stays 1-D (rank-reduced), so the existing
// vector.transfer_write (1 index) remains well-formed.
// CHECK-LABEL: @rank_reducing_subview_writer
// CHECK-SAME: (%[[DST:.*]]: memref<64x64xf32>)
// CHECK-NOT: memref.alloc()
// CHECK: scf.for
// CHECK:   %[[SV:.*]] = memref.subview %[[DST]][%{{.*}}, 0] [1, 64] [1, 1] : memref<64x64xf32> to memref<64xf32, strided<[1], offset: ?>>
// CHECK:   vector.transfer_write %{{.*}}, %[[SV]][%{{.*}}]
// CHECK-NOT: memref.copy
// CHECK: return
