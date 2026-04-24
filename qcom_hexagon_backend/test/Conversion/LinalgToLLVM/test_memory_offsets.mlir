// RUN: linalg-hexagon-opt %s -memory-offsets -split-input-file 2>/dev/null | FileCheck %s

// ===================================================================
// Case 1: Fallback mode — two memref.alloc (space 1) + one
// hexagonmem.alloc (space 0, untouched). Expect: one consolidated
// hexagonmem.alloc sized exactly to need + two memref.view +
// no memref.dealloc.
// ===================================================================
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
// CHECK: %[[GLOBAL_MEM:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<2112xi8, 1>
// CHECK: %[[OFFSET_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[INPUT1_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_ZERO]]][] : memref<2112xi8, 1> to memref<64xi8, 1>
// CHECK: %[[INPUT2_MEM:.*]] = hexagonmem.alloc() : memref<64xi8>
// CHECK: %[[OFFSET_OUT:.*]] = arith.constant 2048 : index
// CHECK: %[[OUTPUT_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_OUT]]][] : memref<2112xi8, 1> to memref<64xi8, 1>
// CHECK: hexagonmem.copy %arg0, %[[INPUT1_VIEW]] : memref<64xi8> to memref<64xi8, 1>
// CHECK: memref.copy %arg1, %[[INPUT2_MEM]] : memref<64xi8> to memref<64xi8>
// CHECK: linalg.add ins(%[[INPUT1_VIEW]], %[[INPUT2_MEM]] : memref<64xi8, 1>, memref<64xi8>) outs(%[[OUTPUT_VIEW]] : memref<64xi8, 1>)
// CHECK: hexagonmem.copy %[[OUTPUT_VIEW]], %arg2 : memref<64xi8, 1> to memref<64xi8>
// CHECK-NOT: memref.dealloc

// -----

// ===================================================================
// Case 2: Fallback mode — tiled loop with two memref.alloc (space 1).
// Expect: one consolidated hexagonmem.alloc + two memref.view hoisted
// above the loop + no memref.dealloc.
// ===================================================================
// C = A + B, length 256, tiled by 64; compute in VTCM (memspace 1).
func.func @add_tiled64_static(%A: memref<256xi8>, %B: memref<256xi8>, %C: memref<256xi8>) {
  %c0   = arith.constant 0   : index
  %c64  = arith.constant 64  : index
  %c256 = arith.constant 256 : index

  %vtA = memref.alloc() : memref<64xi8, 1>
  %vtB = memref.alloc() : memref<64xi8, 1>

  // t = 0, 64, 128, 192
  scf.for %t = %c0 to %c256 step %c64 {
    %aT = memref.subview %A[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    %bT = memref.subview %B[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
    %cT = memref.subview %C[%t] [64] [1]
            : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>

    // Stage inputs to VTCM.
    memref.copy %aT, %vtA
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
    memref.copy %bT, %vtB
      : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>

    // In-place add: vtA = vtA + vtB.
    linalg.add
      ins(%vtA, %vtB : memref<64xi8, 1>, memref<64xi8, 1>)
      outs(%vtA : memref<64xi8, 1>)

    // Write back to C[t : t+64).
    memref.copy %vtA, %cT
      : memref<64xi8, 1> to memref<64xi8, strided<[1], offset: ?>>
  }

  memref.dealloc %vtA : memref<64xi8, 1>
  memref.dealloc %vtB : memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @add_tiled64_static(
// CHECK-SAME:    %arg0: memref<256xi8>, %arg1: memref<256xi8>, %arg2: memref<256xi8>

// Buffer allocation + constants
// CHECK: %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<2112xi8, 1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index

// Views at offsets 0 and 2048
// CHECK: %[[VIEW0:.*]] = memref.view %[[BUF]][%[[C0_0]]][]
// CHECK-SAME: : memref<2112xi8, 1> to memref<64xi8, 1>
// CHECK: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK: %[[VIEW1:.*]] = memref.view %[[BUF]][%[[C2048]]][]
// CHECK-SAME: : memref<2112xi8, 1> to memref<64xi8, 1>

// Loop and tiled subviews
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
// CHECK:   %[[S0:.*]] = memref.subview %arg0[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
// CHECK:   %[[S1:.*]] = memref.subview %arg1[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>
// CHECK:   %[[S2:.*]] = memref.subview %arg2[%[[IV]]] [64] [1] : memref<256xi8> to memref<64xi8, strided<[1], offset: ?>>

// Copies and add
// CHECK:   memref.copy %[[S0]], %[[VIEW0]] : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
// CHECK:   memref.copy %[[S1]], %[[VIEW1]] : memref<64xi8, strided<[1], offset: ?>> to memref<64xi8, 1>
// CHECK:   linalg.add ins(%[[VIEW0]], %[[VIEW1]] : memref<64xi8, 1>, memref<64xi8, 1>) outs(%[[VIEW0]] : memref<64xi8, 1>)
// CHECK:   memref.copy %[[VIEW0]], %[[S2]] : memref<64xi8, 1> to memref<64xi8, strided<[1], offset: ?>>
// CHECK: }
// CHECK-NOT: memref.dealloc

// -----

// ===================================================================
// Case 3: Fallback mode — two memref.alloc (space 1), no scratch arg.
// Expect: one hexagonmem.alloc + two memref.view + no memref.dealloc.
// ===================================================================
func.func @test_fallback_memref_alloc(%A: memref<32xf32>) {
  %0 = memref.alloc() : memref<32xf32, 1>
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_fallback_memref_alloc(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<2176xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VIEW0:.*]] = memref.view %[[BUF]][%[[C0]]][] : memref<2176xi8, 1> to memref<32xf32, 1>
// CHECK:         %[[C2048:.*]] = arith.constant 2048 : index
// CHECK:         %[[VIEW1:.*]] = memref.view %[[BUF]][%[[C2048]]][] : memref<2176xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc
// CHECK-NOT:     memref.alloc

// -----

// ===================================================================
// Case 4: External VTCM mode — vtcm_scratch arg present (static size),
// two memref.alloc (space 1). Expect: two memref.view into the arg,
// no hexagonmem.alloc, no memref.dealloc.
// ===================================================================
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
// CHECK:         %[[C2048:.*]] = arith.constant 2048 : index
// CHECK:         memref.view %arg1[%[[C2048]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// ===================================================================
// Case 5: External VTCM mode — single memref.alloc (space 1) with
// static vtcm_scratch arg. Expect: one memref.view into the arg, no
// hexagonmem.alloc, no memref.dealloc.
// ===================================================================
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

// ===================================================================
// Case 6: Fallback mode — two hexagonmem.alloc (space 1), no scratch
// arg. Expect: one consolidated hexagonmem.alloc + two memref.view +
// no hexagonmem.dealloc.
// ===================================================================
func.func @test_hexagonmem_alloc_passthrough(%A: memref<64xi8>) {
  %0 = hexagonmem.alloc() : memref<64xi8, 1>
  %1 = hexagonmem.alloc() : memref<64xi8, 1>
  hexagonmem.copy %A, %0 : memref<64xi8> to memref<64xi8, 1>
  return
}

// CHECK-LABEL: func.func @test_hexagonmem_alloc_passthrough(
// CHECK:         %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<2112xi8, 1>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %[[BUF]][%[[C0]]][] : memref<2112xi8, 1> to memref<64xi8, 1>
// CHECK:         %[[C2048:.*]] = arith.constant 2048 : index
// CHECK:         memref.view %[[BUF]][%[[C2048]]][] : memref<2112xi8, 1> to memref<64xi8, 1>
// CHECK-NOT:     hexagonmem.dealloc

// -----

// ===================================================================
// Case 7: External VTCM mode — mixed memref.alloc and hexagonmem.alloc
// (both space 1) with static vtcm_scratch arg. Both should be replaced
// with views into the external arg.
// ===================================================================
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
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.view %arg1[%[[C0]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK:         %[[C2048:.*]] = arith.constant 2048 : index
// CHECK:         memref.view %arg1[%[[C2048]]][] : memref<4194304xi8, 1> to memref<32xf32, 1>
// CHECK-NOT:     memref.dealloc

// -----

// ===================================================================
// Case 8: Invalid external scratch type — memref<?xi16, 1> is not a
// valid scratch arg (must be i8). Pass should warn and fall back to
// internal allocation. No views into %arg1.
// ===================================================================
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

// ===================================================================
// Case 9: Dynamic VTCM alloc — not collected by the pass (dynamic
// shape). Its dealloc must be preserved.
// ===================================================================
func.func @test_preserve_dynamic_alloc_dealloc(%n: index) {
  %0 = memref.alloc(%n) : memref<?xf32, 1>
  memref.dealloc %0 : memref<?xf32, 1>
  return
}

// CHECK-LABEL: func.func @test_preserve_dynamic_alloc_dealloc(
// CHECK:         %[[A:.*]] = memref.alloc(%arg0) : memref<?xf32, 1>
// CHECK:         memref.dealloc %[[A]] : memref<?xf32, 1>

// -----

// ===================================================================
// Case 10: Non-identity layout memref in VTCM space — should be
// preserved (not replaced with memref.view). Only identity-layout
// allocs are eligible for offset consolidation.
// ===================================================================
#map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @test_nonidentity_layout_preserved
// CHECK-SAME:  %[[SCRATCH:.*]]: memref<4194304xi8, 1> {hexagon.scratch}
// The non-identity alloc must be preserved as-is:
// CHECK: memref.alloc() : memref<16x16xf32, #map, 1>
// The identity alloc should be replaced with a view into scratch:
// CHECK: memref.view %[[SCRATCH]]
// The non-identity dealloc must be preserved:
// CHECK: memref.dealloc {{.*}} : memref<16x16xf32, #map, 1>
// No identity alloc should remain:
// CHECK-NOT: memref.alloc() : memref<32xf32, 1>
func.func @test_nonidentity_layout_preserved(
    %scratch: memref<4194304xi8, 1> {hexagon.scratch}) {
  // Non-identity layout: transposed — should be skipped by collectAllocInfos.
  %0 = memref.alloc() : memref<16x16xf32, #map, 1>
  // Identity layout: should be replaced with memref.view into scratch.
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.dealloc %0 : memref<16x16xf32, #map, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}
