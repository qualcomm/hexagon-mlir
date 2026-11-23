//===- memory_alloc_optimize.mlir -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Test copy canonicalization with chained memory copies
// Memory spaces: 0 = DDR, 1 = VTCM

// -----

// Test 1: Same memory space chain (DDR -> DDR -> DDR)
func.func @same_space_chain() -> memref<1024xi8> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>

  memref.copy %src, %intermediate : memref<1024xi8> to memref<1024xi8>
  memref.copy %intermediate, %dst : memref<1024xi8> to memref<1024xi8>
  return %dst : memref<1024xi8>
}

// CHECK-LABEL: @same_space_chain
// CHECK-NEXT:  %[[SRC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
// CHECK-NEXT:  return %[[SRC]] : memref<1024xi8>

// -----

// Test 2: Different memory space chain (DDR -> VTCM -> DDR)
func.func @different_space_chain() -> memref<512xf16> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<512xf16, 1>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<512xf16>

  memref.copy %src, %intermediate : memref<512xf16> to memref<512xf16, 1>
  memref.copy %intermediate, %dst : memref<512xf16, 1> to memref<512xf16>
  return %dst : memref<512xf16>
}

// CHECK-LABEL: @different_space_chain
// CHECK-NEXT:  %[[SRC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
// CHECK-NEXT:  return %[[SRC]] : memref<512xf16>

// -----

// Test 3: Multiple readers from intermediate buffer
func.func @multiple_readers() -> (memref<256xf32>, memref<256xf32>) {
  %src = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<256xf32, 1>
  %dst1 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %dst2 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>

  memref.copy %src, %intermediate : memref<256xf32> to memref<256xf32, 1>
  memref.copy %intermediate, %dst1 : memref<256xf32, 1> to memref<256xf32>
  memref.copy %intermediate, %dst2 : memref<256xf32, 1> to memref<256xf32>
  return %dst1, %dst2 : memref<256xf32>, memref<256xf32>
}

// CHECK-LABEL: @multiple_readers
// CHECK-NEXT:  %[[SRC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
// CHECK-NOT:   memref.copy
// CHECK:       return %[[SRC]], %[[SRC]] : memref<256xf32>, memref<256xf32>


// -----

// Test 4: Non-copy use prevents optimization (negative test)
func.func @non_copy_use() -> memref<256xf32> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<256xf32, 1>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %cst = arith.constant 0.0 : f32

  memref.copy %src, %intermediate : memref<256xf32> to memref<256xf32, 1>
  linalg.fill ins(%cst : f32) outs(%intermediate : memref<256xf32, 1>)
  memref.copy %intermediate, %dst : memref<256xf32, 1> to memref<256xf32>
  return %dst : memref<256xf32>
}

// CHECK-LABEL: @non_copy_use
// CHECK-NEXT:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:  %[[SRC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
// CHECK-NEXT:  %[[INT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<256xf32, 1>
// CHECK-NEXT:  %[[DST:.*]] = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
// CHECK-NEXT:  memref.copy %[[SRC]], %[[INT]] : memref<256xf32> to memref<256xf32, 1>
// CHECK-NEXT:  linalg.fill ins(%[[CST]] : f32) outs(%[[INT]] : memref<256xf32, 1>)
// CHECK-NEXT:  memref.copy %[[INT]], %[[DST]] : memref<256xf32, 1> to memref<256xf32>
// CHECK-NEXT:  return %[[DST]] : memref<256xf32>

// -----

// Test 5: AddCopyBeforeYieldPattern with intermediate VTCM buffer and copy-back
func.func @add_copy_before_yield_complex_test(%arg0: memref<10xf32>) -> memref<10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32

  // Initialize the input buffer to something
  linalg.fill ins(%cst_0 : f32) outs(%arg0 : memref<10xf32>)

  // Loop-carried variable is %arg0
  %res_buffer = scf.for %i = %c0 to %c10 step %c1 iter_args(%loop_arg = %arg0) -> (memref<10xf32>) {
    // 1. Copy from %arg0 (DDR) to a VTCM buffer
    %vtcm_buffer = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1>
    memref.copy %loop_arg, %vtcm_buffer : memref<10xf32> to memref<10xf32, 1>

    // 2. Fill the VTCM buffer
    linalg.fill ins(%cst_1 : f32) outs(%vtcm_buffer : memref<10xf32, 1>)

    // 3. Copy VTCM buffer back to an intermediate DDR buffer
    %intermediate_ddr_buffer = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    memref.copy %vtcm_buffer, %intermediate_ddr_buffer : memref<10xf32, 1> to memref<10xf32>
    
    // 4. Yield the intermediate_ddr_buffer. This should trigger the AddCopyBeforeYieldPattern.
    scf.yield %intermediate_ddr_buffer : memref<10xf32>
  }

  return %res_buffer : memref<10xf32>
}

// CHECK-LABEL: @add_copy_before_yield_complex_test
// CHECK-SAME:  (%arg0: memref<10xf32>)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:       linalg.fill ins(%[[CST0]] : f32) outs(%arg0 : memref<10xf32>)
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK-NEXT:    %[[VTCM:.*]] = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1>
// CHECK-NEXT:    memref.copy %arg0, %[[VTCM]] : memref<10xf32> to memref<10xf32, 1>
// CHECK-NEXT:    linalg.fill ins(%[[CST1]] : f32) outs(%[[VTCM]] : memref<10xf32, 1>)
// CHECK-NEXT:    memref.copy %[[VTCM]], %arg0 : memref<10xf32, 1> to memref<10xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %arg0 : memref<10xf32>

// -----

// Test: CopyForwardingPattern — basic VTCM snapshot elimination.
// When a fresh alloc is copied from a source that is never written after the
// copy, and the destination is never written at all, the destination is a pure
// snapshot. All uses of the destination are replaced with the source and the
// copy + alloc are erased.
func.func @copy_forwarding_basic(%src: memref<512xf16, 1>) -> memref<512xf16, 1> {
  %dst = memref.alloc() {alignment = 64 : i64} : memref<512xf16, 1>
  memref.copy %src, %dst : memref<512xf16, 1> to memref<512xf16, 1>
  return %dst : memref<512xf16, 1>
}

// CHECK-LABEL: @copy_forwarding_basic
// CHECK-NEXT:  return %arg0 : memref<512xf16, 1>
// CHECK-NOT:   memref.alloc
// CHECK-NOT:   memref.copy

// -----

// Test: CopyForwardingPattern — no elimination when types differ.
// Source and destination have different memory spaces; direct substitution
// would be unsafe.
func.func @copy_forwarding_type_mismatch(%src: memref<512xf16>) -> memref<512xf16, 1> {
  %dst = memref.alloc() {alignment = 64 : i64} : memref<512xf16, 1>
  memref.copy %src, %dst : memref<512xf16> to memref<512xf16, 1>
  return %dst : memref<512xf16, 1>
}

// CHECK-LABEL: @copy_forwarding_type_mismatch
// CHECK-NEXT:  %[[DST:.*]] = memref.alloc() {alignment = 64 : i64} : memref<512xf16, 1>
// CHECK-NEXT:  memref.copy %arg0, %[[DST]] : memref<512xf16> to memref<512xf16, 1>
// CHECK-NEXT:  return %[[DST]] : memref<512xf16, 1>

// -----

// Test: CopyForwardingPattern — no elimination when dst is passed as a
// scf.for init_arg and the corresponding region iter-arg is written inside
// the loop.
func.func @copy_forwarding_no_elim_dst_iter_arg_written(
    %zero: memref<4xf32>, %result: memref<4xf32>,
    %lb: index, %ub: index, %step: index,
    %lb2: index, %ub2: index)
    -> memref<4xf32> {
  %accum = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  scf.for %i = %lb to %ub step %step {
    memref.copy %zero, %accum : memref<4xf32> to memref<4xf32>
    %9 = scf.for %j = %lb2 to %ub2 step %step iter_args(%a = %accum) -> (memref<4xf32>) {
      memref.copy %result, %a : memref<4xf32> to memref<4xf32>
      scf.yield %a : memref<4xf32>
    }
  }
  return %accum : memref<4xf32>
}

// CHECK-LABEL: @copy_forwarding_no_elim_dst_iter_arg_written
// CHECK-SAME:  (%arg0: memref<4xf32>, %arg1: memref<4xf32>,
// CHECK-SAME:   %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index)
// CHECK-NEXT:  %[[ACCUM:.*]] = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
// CHECK-NEXT:  scf.for %{{.*}} = %arg2 to %arg3 step %arg4 {
// CHECK-NEXT:    memref.copy %arg0, %[[ACCUM]] : memref<4xf32> to memref<4xf32>
// CHECK-NEXT:    scf.for %{{.*}} = %arg5 to %arg6 step %arg4 {
// CHECK-NEXT:      memref.copy %arg1, %[[ACCUM]] : memref<4xf32> to memref<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[ACCUM]] : memref<4xf32>

// -----

// Test: CopyForwardingPattern — no elimination when src is passed as a
// scf.for init_arg after the copy and the corresponding region iter-arg is
// written inside the loop.
func.func @copy_forwarding_no_elim_src_iter_arg_written(
    %src: memref<4xf32, 1>,
    %data: memref<4xf32, 1>,
    %lb: index, %ub: index, %step: index)
    -> memref<4xf32, 1> {
  %dst = memref.alloc() {alignment = 128 : i64} : memref<4xf32, 1>
  memref.copy %src, %dst : memref<4xf32, 1> to memref<4xf32, 1>
  scf.for %j = %lb to %ub step %step iter_args(%s = %src) -> (memref<4xf32, 1>) {
    memref.copy %data, %s : memref<4xf32, 1> to memref<4xf32, 1>
    scf.yield %s : memref<4xf32, 1>
  }
  return %dst : memref<4xf32, 1>
}

// CHECK-LABEL: @copy_forwarding_no_elim_src_iter_arg_written
// CHECK-SAME:  (%arg0: memref<4xf32, 1>, %arg1: memref<4xf32, 1>,
// CHECK-SAME:   %arg2: index, %arg3: index, %arg4: index)
// CHECK-NEXT:  %[[DST:.*]] = memref.alloc() {alignment = 128 : i64} : memref<4xf32, 1>
// CHECK-NEXT:  memref.copy %arg0, %[[DST]] : memref<4xf32, 1> to memref<4xf32, 1>
// CHECK-NEXT:  scf.for %{{.*}} = %arg2 to %arg3 step %arg4 {
// CHECK-NEXT:    memref.copy %arg1, %arg0 : memref<4xf32, 1> to memref<4xf32, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[DST]] : memref<4xf32, 1>

// -----

// Test: CopyForwardingPattern — elimination IS safe when src is passed as a
// scf.for init_arg but the corresponding region iter-arg is never written
// (read-only use).
func.func @copy_forwarding_ok_src_iter_arg_readonly(
    %src: memref<4xf32, 1>,
    %lb: index, %ub: index, %step: index)
    -> (memref<4xf32, 1>, memref<4xf32, 1>) {
  %dst = memref.alloc() {alignment = 128 : i64} : memref<4xf32, 1>
  memref.copy %src, %dst : memref<4xf32, 1> to memref<4xf32, 1>
  %s_out = scf.for %j = %lb to %ub step %step iter_args(%s = %src) -> (memref<4xf32, 1>) {
    scf.yield %s : memref<4xf32, 1>
  }
  return %dst, %s_out : memref<4xf32, 1>, memref<4xf32, 1>
}

// CHECK-LABEL: @copy_forwarding_ok_src_iter_arg_readonly
// CHECK-NEXT:  return %arg0, %arg0 : memref<4xf32, 1>, memref<4xf32, 1>
// CHECK-NOT:   memref.alloc
// CHECK-NOT:   memref.copy
