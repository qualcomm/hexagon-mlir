//===- duplicate-copy-elimination.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Tests for DuplicateCopyEliminationPattern:
// Two memref.copy ops in the same block with the same source and identical
// destination type (both targeting fresh allocs) — the first copy is
// redundant when its destination has no other uses, and is eliminated.

// -----

// Test 1: Basic duplicate copy — first copy and its alloc are eliminated,
// uses replaced with the second destination.
func.func @basic_duplicate_copy(%src: memref<128xf16, 1>) -> (memref<128xf16, 1>, memref<128xf16, 1>) {
  %a = memref.alloc() {alignment = 128 : i64} : memref<128xf16, 1>
  %b = memref.alloc() {alignment = 128 : i64} : memref<128xf16, 1>
  memref.copy %src, %a : memref<128xf16, 1> to memref<128xf16, 1>
  memref.copy %src, %b : memref<128xf16, 1> to memref<128xf16, 1>
  return %b, %b : memref<128xf16, 1>, memref<128xf16, 1>
}

// CHECK-LABEL: @basic_duplicate_copy
// CHECK-NOT:   memref.alloc()
// CHECK-NOT:   memref.copy
// CHECK:       return %arg0, %arg0

// -----

// Test 2: Duplicate copy inside an scf.for body — models the VTCMTiling
// pattern where the same DDR subview is copied into two VTCM buffers per
// iteration. The first copy is eliminated and %a is replaced by %b.
func.func @duplicate_copy_in_loop(%src: memref<1024x128xf16>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c1024 step %c128 {
    %subview = memref.subview %src[%iv, 0] [128, 128] [1, 1]
                : memref<1024x128xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %a = memref.alloc() {alignment = 128 : i64} : memref<128x128xf16, 1>
    %b = memref.alloc() {alignment = 128 : i64} : memref<128x128xf16, 1>
    memref.copy %subview, %a : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, 1>
    memref.copy %subview, %b : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, 1>
  }
  return
}

// CHECK-LABEL: @duplicate_copy_in_loop
// CHECK:       scf.for
// CHECK:         %[[B:.*]] = memref.alloc()
// CHECK:         memref.copy %{{.*}}, %[[B]]
// CHECK-NOT:     memref.alloc()

// -----

// Test 3: Intervening write to src between the two copies — must NOT eliminate
// the first copy (negative test for noWritesBetween guard).
func.func @intervening_write_to_src(%src: memref<64xf32, 1>, %other: memref<64xf32, 1>) -> (memref<64xf32, 1>, memref<64xf32, 1>) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  %b = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  memref.copy %src, %a : memref<64xf32, 1> to memref<64xf32, 1>
  // This copy writes to %src, so the second copy below sees a different value.
  memref.copy %other, %src : memref<64xf32, 1> to memref<64xf32, 1>
  memref.copy %src, %b : memref<64xf32, 1> to memref<64xf32, 1>
  return %a, %b : memref<64xf32, 1>, memref<64xf32, 1>
}

// CHECK-LABEL: @intervening_write_to_src
// CHECK:       %[[A:.*]] = memref.alloc()
// CHECK:       memref.copy %{{.*}}, %[[A]]
// CHECK:       memref.copy %{{.*}}, %arg0
// CHECK:       return %[[A]], %arg0

// -----

// Test 4: Prior destination has other uses — must NOT eliminate
// (negative test for hasOneUse guard).
func.func @prior_dst_has_other_uses(%src: memref<64xf32, 1>) -> (memref<64xf32, 1>, memref<64xf32, 1>) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  %b = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  memref.copy %src, %a : memref<64xf32, 1> to memref<64xf32, 1>
  memref.copy %src, %b : memref<64xf32, 1> to memref<64xf32, 1>
  // %a is used in return — priorDst has more than one use, so not eliminated.
  return %a, %b : memref<64xf32, 1>, memref<64xf32, 1>
}

// CHECK-LABEL: @prior_dst_has_other_uses
// CHECK-NOT:   memref.alloc()
// CHECK-NOT:   memref.copy
// CHECK:       return %arg0, %arg0

// -----

// Test 5: Type mismatch between the two destination buffers — must NOT
// eliminate (different destination types, so dstType comparison fails).
// Both copies use the same source to ensure it is the destination type guard
// that prevents elimination, not a source mismatch.
func.func @type_mismatch_no_elim(%src: memref<64xf32>) -> (memref<64xf32, 1>, memref<64xf32>) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<64xf32, 1>
  %b = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %a : memref<64xf32> to memref<64xf32, 1>
  memref.copy %src, %b : memref<64xf32> to memref<64xf32>
  return %a, %b : memref<64xf32, 1>, memref<64xf32>
}

// CHECK-LABEL: @type_mismatch_no_elim
// CHECK:       %[[A:.*]] = memref.alloc() {{.*}} : memref<64xf32, 1>
// CHECK:       memref.copy %{{.*}}, %[[A]]
// CHECK:       return %[[A]], %arg0

// -----

// Test 6: Three copies from the same source — first and second are
// eliminated (each has only one use), collapsing to the third destination.
func.func @triple_duplicate_copy(%src: memref<32xf32, 1>) -> (memref<32xf32, 1>, memref<32xf32, 1>, memref<32xf32, 1>) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
  %b = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
  %c = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
  memref.copy %src, %a : memref<32xf32, 1> to memref<32xf32, 1>
  memref.copy %src, %b : memref<32xf32, 1> to memref<32xf32, 1>
  memref.copy %src, %c : memref<32xf32, 1> to memref<32xf32, 1>
  return %c, %c, %c : memref<32xf32, 1>, memref<32xf32, 1>, memref<32xf32, 1>
}

// CHECK-LABEL: @triple_duplicate_copy
// CHECK-NOT:   memref.alloc()
// CHECK-NOT:   memref.copy
// CHECK:       return %arg0, %arg0, %arg0
