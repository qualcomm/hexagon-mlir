//===- copy-canonicalization-func-calls.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Tests for the interprocedural WrittenArgsSet extension to
// CopyForwardingPattern and DuplicateCopyEliminationPattern, and for the
// extended alias-chain tracking in hasWriteAfter (expand_shape, collapse_shape,
// reinterpret_cast, extract_strided_metadata).
//
// Key properties exercised:
//   1. A callee that WRITES to a memref argument prevents copy-forwarding.
//   2. A read-only callee allows copy-forwarding.
//   3. A write callee on the prior dst prevents duplicate-copy elimination.
//   4-8. Writes through each alias op (expand_shape, collapse_shape,
//        reinterpret_cast, extract_strided_metadata, cast) are detected and
//        prevent copy-forwarding.

// -----

// Test 1: callee writes to its arg -- CopyForwardingPattern must NOT eliminate
// the alloc+copy because dst has a future write via the call.
func.func @callee_write(%buf: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %v = arith.constant dense<1.000000e+00> : vector<64xf32>
  vector.transfer_write %v, %buf[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
  return
}

func.func @copy_forwarding_blocked_by_write_callee(%src: memref<64xf32>) -> memref<64xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %alloc : memref<64xf32> to memref<64xf32>
  func.call @callee_write(%alloc) : (memref<64xf32>) -> ()
  return %alloc : memref<64xf32>
}

// The alloc and copy must survive because the callee overwrites %alloc after
// the copy -- forwarding src directly would change semantics.
// CHECK-LABEL: @copy_forwarding_blocked_by_write_callee
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: call @callee_write(%[[ALLOC]])
// CHECK: return %[[ALLOC]]

// -----

// Test 2: callee only reads its arg -- CopyForwardingPattern SHOULD eliminate
// the alloc+copy and forward the source directly to the callee.
func.func @callee_read(%buf: memref<64xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %r = vector.reduction <add>, %v : vector<64xf32> into f32
  return %r : f32
}

func.func @copy_forwarding_allowed_by_read_only_callee(%src: memref<64xf32>) -> f32 {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %alloc : memref<64xf32> to memref<64xf32>
  %r = func.call @callee_read(%alloc) : (memref<64xf32>) -> f32
  return %r : f32
}

// The alloc and copy are eliminated; %src is forwarded directly to callee_read.
// CHECK-LABEL: @copy_forwarding_allowed_by_read_only_callee
// CHECK-NOT: memref.alloc()
// CHECK-NOT: memref.copy
// CHECK: %[[R:.*]] = call @callee_read(%arg0)
// CHECK: return %[[R]]

// -----

// Test 3: Two copies from the same source; the first destination is passed to
// a write callee. DuplicateCopyEliminationPattern must NOT merge the two allocs
// because the prior destination is written after its initialising copy.
func.func @accumulate(%buf: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %v = arith.constant dense<2.000000e+00> : vector<64xf32>
  vector.transfer_write %v, %buf[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
  return
}

func.func @dup_copy_elim_blocked_by_write_callee(%src: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  %b = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %a : memref<64xf32> to memref<64xf32>
  memref.copy %src, %b : memref<64xf32> to memref<64xf32>
  func.call @accumulate(%a) : (memref<64xf32>) -> ()
  return %a, %b : memref<64xf32>, memref<64xf32>
}

// Both allocs and both copies must survive: %a (prior dst) gets written by the
// call, so merging %b -> %a would corrupt %b's snapshot of %src.
// CHECK-LABEL: @dup_copy_elim_blocked_by_write_callee
// CHECK: %[[A:.*]] = memref.alloc()
// CHECK: %[[B:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[A]]
// CHECK: memref.copy %{{.*}}, %[[B]]
// CHECK: call @accumulate(%[[A]])
// CHECK: return %[[A]], %[[B]]

// -----

// Test 4: write via memref.expand_shape alias — copy forwarding must be blocked.
func.func @write_via_expand_shape(%src: memref<64xf32>) -> memref<64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : vector<8x8xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %alloc : memref<64xf32> to memref<64xf32>
  %exp = memref.expand_shape %alloc [[0, 1]] output_shape [8, 8]
      : memref<64xf32> into memref<8x8xf32>
  vector.transfer_write %cst, %exp[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x8xf32>, memref<8x8xf32>
  return %alloc : memref<64xf32>
}

// CHECK-LABEL: @write_via_expand_shape
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: memref.expand_shape %[[ALLOC]]
// CHECK: vector.transfer_write
// CHECK: return %[[ALLOC]]

// -----

// Test 5: write via memref.collapse_shape alias — copy forwarding must be blocked.
func.func @write_via_collapse_shape(%src: memref<8x8xf32>) -> memref<8x8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<2.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  memref.copy %src, %alloc : memref<8x8xf32> to memref<8x8xf32>
  %col = memref.collapse_shape %alloc [[0, 1]]
      : memref<8x8xf32> into memref<64xf32>
  vector.transfer_write %cst, %col[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32>
  return %alloc : memref<8x8xf32>
}

// CHECK-LABEL: @write_via_collapse_shape
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: memref.collapse_shape %[[ALLOC]]
// CHECK: vector.transfer_write
// CHECK: return %[[ALLOC]]

// -----

// Test 6: write via memref.reinterpret_cast alias — copy forwarding must be blocked.
func.func @write_via_reinterpret_cast(%src: memref<64xf32>) -> memref<64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<3.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %alloc : memref<64xf32> to memref<64xf32>
  %rc = memref.reinterpret_cast %alloc to offset: [0], sizes: [64], strides: [1]
      : memref<64xf32> to memref<64xf32, strided<[1]>>
  vector.transfer_write %cst, %rc[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32, strided<[1]>>
  return %alloc : memref<64xf32>
}

// CHECK-LABEL: @write_via_reinterpret_cast
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: memref.reinterpret_cast %[[ALLOC]]
// CHECK: vector.transfer_write
// CHECK: return %[[ALLOC]]

// -----

// Test 7: write via memref.extract_strided_metadata base alias — copy
// forwarding must be blocked.  extract_strided_metadata result(0) is the
// base-buffer pointer, which aliases the original alloc; a write through a
// reinterpret_cast of that base reaches the alloc's storage.
func.func @write_via_extract_strided_metadata(%src: memref<64xf32>) -> memref<64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<4.000000e+00> : vector<64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  memref.copy %src, %alloc : memref<64xf32> to memref<64xf32>
  %base, %off, %sz, %st =
      memref.extract_strided_metadata %alloc
          : memref<64xf32> -> memref<f32>, index, index, index
  %rc = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [1]
      : memref<f32> to memref<64xf32>
  vector.transfer_write %cst, %rc[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32>
  return %alloc : memref<64xf32>
}

// CHECK-LABEL: @write_via_extract_strided_metadata
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: vector.transfer_write
// CHECK: return %[[ALLOC]]
