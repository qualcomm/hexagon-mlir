//===- copy_from_zero_filled_alloc.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Tests for CopyFromZeroFilledAllocPattern.  When `memref.copy %src, %dst`
// has `%src` defined by an alloc whose only writers are zero-fill ops
// (vector.transfer_write of a zero splat or linalg.fill ins(0)), the copy
// is rewritten to `linalg.fill ins(0) outs(%dst)` and the staging zero
// alloc is erased once it has no remaining copy users.

// -----

// Test 1: vector.transfer_write of a zero splat is the only writer of the
// alloc, then memref.copy %alloc, %dst.
func.func @vector_zero_then_copy(%dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  return
}

// CHECK-LABEL: @vector_zero_then_copy
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32>)
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.copy
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[DST]] : memref<128xf32>)
// CHECK: return

// -----

// Test 2: linalg.fill ins(0) is the only writer, then memref.copy.
func.func @linalg_fill_zero_then_copy(%dst: memref<64xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  linalg.fill ins(%f0 : f32) outs(%alloc : memref<64xf32>)
  memref.copy %alloc, %dst : memref<64xf32> to memref<64xf32>
  return
}

// CHECK-LABEL: @linalg_fill_zero_then_copy
// CHECK-SAME: (%[[DST:.*]]: memref<64xf32>)
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.copy
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[DST]] : memref<64xf32>)
// CHECK: return

// -----

// Test 3: a single zero alloc feeds two destination copies; both copies are
// rewritten and the alloc is erased after both fire.
func.func @multiple_dst_copies_share_zero_alloc(%dst1: memref<32xf32>, %dst2: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<32xf32>, memref<32xf32>
  memref.copy %alloc, %dst1 : memref<32xf32> to memref<32xf32>
  memref.copy %alloc, %dst2 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: @multiple_dst_copies_share_zero_alloc
// CHECK-SAME: (%[[DST1:.*]]: memref<32xf32>, %[[DST2:.*]]: memref<32xf32>)
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.copy
// CHECK-DAG: linalg.fill ins(%{{.*}} : f32) outs(%[[DST1]] : memref<32xf32>)
// CHECK-DAG: linalg.fill ins(%{{.*}} : f32) outs(%[[DST2]] : memref<32xf32>)
// CHECK: return

// -----

// Test 4: Negative — alloc has a non-zero transfer_write before the copy, so
// CopyFromZeroFilledAllocPattern bails (`isZeroFilledAlloc` rejects the
// non-zero writer).  The alloc is placed in VTCM (memory space 1) draining to
// a DDR (space 0) dst so the sibling RedirectAllocWritesToDestinationPattern —
// which would otherwise redirect both writers to %dst — also bails on its
// memory-hierarchy guard.  The pattern must leave the alloc, the writes and
// the copy untouched.
func.func @negative_nonzero_write_blocks_rewrite(%dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %cz = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cnz = arith.constant dense<7.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
  vector.transfer_write %cz, %alloc[%c0] : vector<128xf32>, memref<128xf32, 1>
  vector.transfer_write %cnz, %alloc[%c0] : vector<128xf32>, memref<128xf32, 1>
  memref.copy %alloc, %dst : memref<128xf32, 1> to memref<128xf32>
  return
}

// CHECK-LABEL: @negative_nonzero_write_blocks_rewrite
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<128xf32, 1>
// CHECK: vector.transfer_write
// CHECK: vector.transfer_write
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK-NOT: linalg.fill
// CHECK: return

// -----

// Test 5: Negative — alloc is read by a transfer_read.  The pattern must
// not redirect the copy to the destination because the read sees the
// staging buffer's contents.
func.func @negative_alloc_is_read(%dst: memref<128xf32>) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  vector.transfer_write %cst, %alloc[%c0] : vector<128xf32>, memref<128xf32>
  %read = vector.transfer_read %alloc[%c0], %f0 : memref<128xf32>, vector<128xf32>
  memref.copy %alloc, %dst : memref<128xf32> to memref<128xf32>
  return %read : vector<128xf32>
}

// CHECK-LABEL: @negative_alloc_is_read
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
// CHECK: memref.copy %[[ALLOC]], %{{.*}}
// CHECK-NOT: linalg.fill
// CHECK: return

// -----

// Test 6: Negative — the alloc's first in-block use is a non-zero
// transfer_write; a zero transfer_write follows.  firstUseIsZeroFill must
// reject because the buffer briefly holds non-zero contents before the
// zero-fill, and a future refactor that drops this guard could silently
// rewrite the copy.  As in test 4 the alloc is VTCM draining to a DDR dst so
// the sibling RedirectAllocWritesToDestinationPattern bails on its
// memory-hierarchy guard rather than redirecting the writers.
func.func @negative_first_use_is_nonzero(%dst: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %cnz = arith.constant dense<7.000000e+00> : vector<128xf32>
  %cz = arith.constant dense<0.000000e+00> : vector<128xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
  vector.transfer_write %cnz, %alloc[%c0] : vector<128xf32>, memref<128xf32, 1>
  vector.transfer_write %cz, %alloc[%c0] : vector<128xf32>, memref<128xf32, 1>
  memref.copy %alloc, %dst : memref<128xf32, 1> to memref<128xf32>
  return
}

// CHECK-LABEL: @negative_first_use_is_nonzero
// CHECK-SAME: (%[[DST:.*]]: memref<128xf32>)
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<128xf32, 1>
// CHECK: vector.transfer_write
// CHECK: vector.transfer_write
// CHECK: memref.copy %[[ALLOC]], %[[DST]]
// CHECK-NOT: linalg.fill {{.*}} outs(%[[DST]]
// CHECK: return

// -----

// Test 7: Negative — the only zero-fill of the alloc lives inside an
// scf.for body, so the buffer is not guaranteed zero on the first iteration
// (and may not run at all for an empty trip count).  isZeroFilledAlloc on
// its own would accept this — every user is a recognized zero-fill or
// reading copy — but firstUseIsZeroFill must reject because the fill's
// block differs from the alloc's defining block.  The alloc is VTCM draining
// to a DDR dst so the sibling RedirectAllocWritesToDestinationPattern — which
// would otherwise redirect the in-loop fill to %dst — bails on its
// memory-hierarchy guard.
func.func @negative_zero_fill_out_of_block(%dst: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    linalg.fill ins(%f0 : f32) outs(%alloc : memref<32xf32, 1>)
  }
  memref.copy %alloc, %dst : memref<32xf32, 1> to memref<32xf32>
  return
}

// CHECK-LABEL: @negative_zero_fill_out_of_block
// CHECK-SAME: (%[[DST:.*]]: memref<32xf32>)
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<32xf32, 1>
// CHECK: scf.for
// CHECK:   linalg.fill {{.*}} outs(%[[ALLOC]]
// CHECK: memref.copy %[[ALLOC]], %[[DST]]
// CHECK-NOT: linalg.fill {{.*}} outs(%[[DST]]
// CHECK: return
