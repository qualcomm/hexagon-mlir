// RUN:linalg-hexagon-opt %s -memory-offsets -split-input-file | FileCheck %s
// ===== Input: Function 1 =====
func.func @add_using_offsets(%arg0: memref<64xi8>, %arg1: memref<64xi8>, %arg2: memref<64xi8>) {
  %0 = hexagonmem.alloc() : memref<64xi8, 1>
  %1 = hexagonmem.alloc() : memref<64xi8>
  %2 = hexagonmem.alloc() : memref<64xi8, 1>
  hexagonmem.copy %arg0, %0 : memref<64xi8> to memref<64xi8, 1>
  memref.copy %arg1, %1 : memref<64xi8> to memref<64xi8>
  linalg.add ins(%0, %1 : memref<64xi8, 1>, memref<64xi8>) outs(%2 : memref<64xi8, 1>)
  hexagonmem.copy %2, %arg2 : memref<64xi8, 1> to memref<64xi8>
  return
}

// ====== CHECKS for @add_using_offsets (only transformation-relevant) ======
// CHECK-LABEL: func.func @add_using_offsets(
// CHECK: %[[GLOBAL_MEM:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1048576xi8, 1>
// CHECK: %[[OFFSET_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[INPUT1_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_ZERO]]][] : memref<1048576xi8, 1> to memref<64xi8, 1>
// CHECK: %[[INPUT2_MEM:.*]] = hexagonmem.alloc() : memref<64xi8>
// CHECK: %[[OFFSET_OUT:.*]] = arith.constant 2048 : index
// CHECK: %[[OUTPUT_VIEW:.*]] = memref.view %[[GLOBAL_MEM]][%[[OFFSET_OUT]]][] : memref<1048576xi8, 1> to memref<64xi8, 1>
// CHECK: hexagonmem.copy %arg0, %[[INPUT1_VIEW]] : memref<64xi8> to memref<64xi8, 1>
// CHECK: memref.copy %arg1, %[[INPUT2_MEM]] : memref<64xi8> to memref<64xi8>
// CHECK: linalg.add ins(%[[INPUT1_VIEW]], %[[INPUT2_MEM]] : memref<64xi8, 1>, memref<64xi8>) outs(%[[OUTPUT_VIEW]] : memref<64xi8, 1>)
// CHECK: hexagonmem.copy %[[OUTPUT_VIEW]], %arg2 : memref<64xi8, 1> to memref<64xi8>

// -----

// ===== Input: Function 2 =====
// C = A + B, length 256, tiled by 64; compute in VTCM (memspace 1).
func.func @add_tiled64_static(%A: memref<256xi8>, %B: memref<256xi8>, %C: memref<256xi8>) {
  %c0   = arith.constant 0   : index
  %c64  = arith.constant 64  : index
  %c256 = arith.constant 256 : index

  %vtA = hexagonmem.alloc() : memref<64xi8, 1>
  %vtB = hexagonmem.alloc() : memref<64xi8, 1>

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

  return
}

// ====== CHECKS for @add_tiled64_static ======
// CHECK-LABEL: func.func @add_tiled64_static(
// CHECK-SAME:    %arg0: memref<256xi8>, %arg1: memref<256xi8>, %arg2: memref<256xi8>

// Buffer allocation + constants
// CHECK: %[[BUF:.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1048576xi8, 1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index

// Views at offsets 0 and 2048
// CHECK: %[[VIEW0:.*]] = memref.view %[[BUF]][%[[C0_0]]][]
// CHECK-SAME: : memref<1048576xi8, 1> to memref<64xi8, 1>
// CHECK: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK: %[[VIEW1:.*]] = memref.view %[[BUF]][%[[C2048]]][]
// CHECK-SAME: : memref<1048576xi8, 1> to memref<64xi8, 1>

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
