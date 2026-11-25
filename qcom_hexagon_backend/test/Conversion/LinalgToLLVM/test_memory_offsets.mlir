//===- test_memory_offsets.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -memory-offsets -split-input-file 2>/dev/null | FileCheck %s

// Case 1: Two VTCM allocs (overlapping) + one DDR alloc (untouched).
// Both VTCM allocs get views; DDR alloc stays; deallocs erased.
func.func @add_using_offsets(%arg0: memref<64xi8>, %arg1: memref<64xi8>, %arg2: memref<64xi8>) {
  %0 = memref.alloc() : memref<64xi8, 1>
  %1 = hexagonmem.alloc() : memref<64xi8>
  %2 = memref.alloc() : memref<64xi8, 1>
  hexagonmem.copy %arg0, %0 : memref<64xi8> to memref<64xi8, 1>
  memref.copy %arg1, %1 : memref<64xi8> to memref<64xi8>
  linalg.add ins(%0, %1 : memref<64xi8, 1>, memref<64xi8>) outs(%2 : memref<64xi8, 1>)
  hexagonmem.copy %2, %arg2 : memref<64xi8, 1> to memref<64xi8>
  memref.dealloc %0 : memref<64xi8, 1>
  memref.dealloc %2 : memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @add_using_offsets(
// CHECK: %[[GLOBAL_MEM:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<192xi8, 1>
// CHECK: %[[OFFSET_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[INPUT1_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_ZERO]]][] : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK: %[[INPUT2_MEM:.*]] = hexagonmem.alloc() : memref<64xi8>
// CHECK: %[[OFFSET_OUT:.*]] = arith.constant 128 : index
// CHECK: %[[OUTPUT_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_OUT]]][] : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK: hexagonmem.copy %arg0, %[[INPUT1_VIEW]] : memref<64xi8> to memref<64xi8, 1>
// CHECK: memref.copy %arg1, %[[INPUT2_MEM]] : memref<64xi8> to memref<64xi8>
// CHECK: linalg.add ins(%[[INPUT1_VIEW]], %[[INPUT2_MEM]] : memref<64xi8, 1>, memref<64xi8>) outs(%[[OUTPUT_VIEW]] : memref<64xi8, 1>)
// CHECK: hexagonmem.copy %[[OUTPUT_VIEW]], %arg2 : memref<64xi8, 1> to memref<64xi8>
// CHECK-NOT: memref.dealloc

// -----

// Case 2: Two VTCM staging buffers used inside a tiled loop.
// Views hoisted to function entry; loop uses views directly.
func.func @add_tiled64_static(%A: memref<256xi8>, %B: memref<256xi8>, %C: memref<256xi8>) {
  %c0   = arith.constant 0   : index
  %c64  = arith.constant 64  : index
  %c256 = arith.constant 256 : index

  %vtA = memref.alloc() : memref<64xi8, 1>
  %vtB = memref.alloc() : memref<64xi8, 1>

  scf.for %t = %c0 to %c256 step %c64 {
    %aT = memref.subview %A[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    %bT = memref.subview %B[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    %cT = memref.subview %C[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    memref.copy %aT, %vtA
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
    memref.copy %bT, %vtB
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
    linalg.add
      ins(%vtA, %vtB : memref<64xi8, 1>, memref<64xi8, 1>)
      outs(%vtA : memref<64xi8, 1>)
    memref.copy %vtA, %cT
      : memref<64xi8, 1> to memref<64xi8, strided<[1], offset: ?>>
  }

  memref.dealloc %vtA : memref<64xi8, 1>
  memref.dealloc %vtB : memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @add_tiled64_static(
// CHECK-SAME:    %arg0: memref<256xi8>, %arg1: memref<256xi8>, %arg2: memref<256xi8>
// CHECK: %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<192xi8, 1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK: %[[VIEW0:.*]] = memref.view %[[BUF]][%[[C0_0]]][]
// CHECK-SAME: : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[VIEW1:.*]] = memref.view %[[BUF]][%[[C128]]][]
// CHECK-SAME: : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
// CHECK:   %[[S0:.*]] = memref.subview %arg0[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
// CHECK:   %[[S1:.*]] = memref.subview %arg1[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
// CHECK:   %[[S2:.*]] = memref.subview %arg2[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
// CHECK:   memref.copy %[[S0]], %[[VIEW0]] : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
// CHECK:   memref.copy %[[S1]], %[[VIEW1]] : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
// CHECK:   linalg.add ins(%[[VIEW0]], %[[VIEW1]] : memref<64xi8, 1>, memref<64xi8, 1>) outs(%[[VIEW0]] : memref<64xi8, 1>)
// CHECK:   memref.copy %[[VIEW0]], %[[S2]] : memref<64xi8, 1> to memref<64xi8, strided<[1], offset: ?>>
// CHECK: }
// CHECK-NOT: memref.dealloc

// -----

// Case 3: Two overlapping VTCM allocs, fallback mode (no scratch arg).
func.func @test_fallback_memref_alloc(%A: memref<32xf32>) {
  %0 = memref.alloc() : memref<32xf32, 1>
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_fallback_memref_alloc(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<256xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VIEW0:.*]] = memref.view %[[BUF]][%[[C0]]][] : memref<256xi8, 1> to memref<32xf32, 1>
// CHECK:         %[[C128:.*]] = arith.constant 128 : index
// CHECK:         %[[VIEW1:.*]] = memref.view %[[BUF]][%[[C128]]][] : memref<256xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc
// CHECK-NOT:     memref.alloc

// -----

// Case 4: External VTCM mode — views into scratch arg, no internal alloc.
func.func @test_external_vtcm_arg(
    %A: memref<32xf32>,
    %vtcm: memref<4194304xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_external_vtcm_arg(
// CHECK-NOT:     hexagonmem.alloc
// CHECK-NOT:     memref.alloc
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %arg1[%[[C0]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK:         %[[C128:.*]] = arith.constant 128 : index
// CHECK:         memref.view %arg1[%[[C128]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// Case 5: Single VTCM alloc, external mode.
func.func @test_single_alloc_external(
    %A: memref<32xf32>,
    %vtcm: memref<4194304xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_single_alloc_external(
// CHECK-NOT:     hexagonmem.alloc
// CHECK-NOT:     memref.alloc
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %arg1[%[[C0]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// Case 6: Two hexagonmem.alloc (VTCM), fallback mode.
func.func @test_hexagonmem_alloc_passthrough(%A: memref<64xi8>) {
  %0 = hexagonmem.alloc() : memref<64xi8, 1>
  %1 = hexagonmem.alloc() : memref<64xi8, 1>
  hexagonmem.copy %A, %0 : memref<64xi8> to memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @test_hexagonmem_alloc_passthrough(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<192xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %[[BUF]][%[[C0]]][] : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK:         %[[C128:.*]] = arith.constant 128 : index
// CHECK:         memref.view %[[BUF]][%[[C128]]][] : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK-NOT:     hexagonmem.dealloc

// -----

// Case 7: Mixed memref.alloc + hexagonmem.alloc (both VTCM), external mode.
func.func @test_mixed_allocs(%A: memref<32xf32>,
                             %vtcm: memref<4194304xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  %1 = hexagonmem.alloc() : memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_mixed_allocs(
// CHECK-NOT:     hexagonmem.alloc
// CHECK-NOT:     memref.alloc
// CHECK-DAG:     memref.view %arg1[%{{.*}}][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-DAG:     memref.view %arg1[%{{.*}}][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// Case 8: Invalid scratch type — warns + falls back to internal alloc.
func.func @test_invalid_external_scratch_type(
    %A: memref<32xf32>,
    %vtcm_bad: memref<?xi16, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_invalid_external_scratch_type(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<128xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %[[BUF]][%[[C0]]][] : memref<128xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.view %arg1
// CHECK-NOT:     memref.dealloc

// -----

// Case 9: Dynamic VTCM alloc — skipped; dealloc preserved.
func.func @test_preserve_dynamic_alloc_dealloc(%n: index) {
  %0 = memref.alloc(%n) : memref<?xf32, 1>
  memref.dealloc %0 : memref<?xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_preserve_dynamic_alloc_dealloc(
// CHECK:         %[[A:.*]] = memref.alloc(%arg0) : memref<?xf32, 1>
// CHECK:         memref.dealloc %[[A]] : memref<?xf32, 1>

// -----

// Case 10: Sequential allocs — non-overlapping lifetimes share offset 0.
// Buffer sized for one alloc, not two.
func.func @test_sequential_reuse(%A: memref<128xf32>) {
  %0 = memref.alloc() : memref<128xf32, 1>
  memref.copy %A, %0 : memref<128xf32> to memref<128xf32, 1>
  memref.dealloc %0 : memref<128xf32, 1>
  %1 = memref.alloc() : memref<128xf32, 1>
  memref.copy %A, %1 : memref<128xf32> to memref<128xf32, 1>
  memref.dealloc %1 : memref<128xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_sequential_reuse(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<512xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[V0:.*]] = memref.view %[[BUF]][%[[C0]]][] : memref<512xi8, 1> to memref<128xf32, 1>
// CHECK:         memref.copy %arg0, %[[V0]]
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[V1:.*]] = memref.view %[[BUF]][%[[C0_1]]][] : memref<512xi8, 1> to memref<128xf32, 1>
// CHECK:         memref.copy %arg0, %[[V1]]
// CHECK-NOT:     memref.dealloc

// -----

// Case 11: Partial overlap — %0/%1 live simultaneously (different offsets);
// %2 reuses %0's offset after %0 is dead.
func.func @test_partial_overlap_reuse(%A: memref<128xf32>, %B: memref<128xf32>) {
  %0 = memref.alloc() : memref<128xf32, 1>
  memref.copy %A, %0 : memref<128xf32> to memref<128xf32, 1>
  %1 = memref.alloc() : memref<128xf32, 1>
  memref.copy %B, %1 : memref<128xf32> to memref<128xf32, 1>
  memref.dealloc %0 : memref<128xf32, 1>
  %2 = memref.alloc() : memref<128xf32, 1>
  memref.copy %A, %2 : memref<128xf32> to memref<128xf32, 1>
  memref.dealloc %1 : memref<128xf32, 1>
  memref.dealloc %2 : memref<128xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_partial_overlap_reuse(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1024xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %[[BUF]][%[[C0]]][] : memref<1024xi8, 1> to memref<128xf32, 1>
// CHECK:         %[[C512:.*]] = arith.constant 512 : index
// CHECK:         memref.view %[[BUF]][%[[C512]]][] : memref<1024xi8, 1> to memref<128xf32, 1>
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         memref.view %[[BUF]][%[[C0_1]]][] : memref<1024xi8, 1> to memref<128xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// Case 12: Non-identity layout skipped; identity layout replaced.
#map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @test_nonidentity_layout_preserved
// CHECK-SAME:  %[[SCRATCH:.*]]: memref<4194304xi8, 1> {hexagon.scratch}
// CHECK: memref.alloc() : memref<16x16xf32, #map, 1>
// CHECK: memref.view %[[SCRATCH]]
// CHECK: memref.dealloc {{.*}} : memref<16x16xf32, #map, 1>
// CHECK-NOT: memref.alloc() : memref<32xf32, 1>
func.func @test_nonidentity_layout_preserved(
    %scratch: memref<4194304xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<16x16xf32, #map, 1>
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.dealloc %0 : memref<16x16xf32, #map, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// -----

// Case 13: Loop-escaped alloc (%vtA, no dealloc) overlaps %vtB across loop.
// Both get distinct offsets.
func.func @test_loop_escaped_alloc(%A: memref<256xi8>, %B: memref<256xi8>) {
  %c0   = arith.constant 0   : index
  %c64  = arith.constant 64  : index
  %c256 = arith.constant 256 : index

  %vtA = memref.alloc() : memref<64xi8, 1>
  %vtB = memref.alloc() : memref<64xi8, 1>

  scf.for %t = %c0 to %c256 step %c64 {
    %aT = memref.subview %A[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    memref.copy %aT, %vtA
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
    memref.copy %aT, %vtB
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
  }

  memref.dealloc %vtB : memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @test_loop_escaped_alloc(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<192xi8, 1>
// CHECK:         memref.view %[[BUF]][%{{.*}}][] : memref<192xi8, 1> to memref<64xi8, 1>
// CHECK:         %[[C128:.*]] = arith.constant 128 : index
// CHECK:         memref.view %[[BUF]][%[[C128]]][] : memref<192xi8, 1> to memref<64xi8, 1>
