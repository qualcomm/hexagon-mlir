// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-double-buffer-generic-s1),cse)' %s | FileCheck %s

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x2048xf32>, %[[ARG1:.*]]: memref<1024x2048xf32>, %[[ARG2:.*]]: memref<1024x2048xf32>)
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1024x2048xf32>
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[TOGGLE:.*]] = memref.alloc() : memref<i1>
// CHECK: memref.store %[[TRUE]], %[[TOGGLE]][] : memref<i1>
//
// CHECK: %[[ALLOC_PING_0:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
// CHECK: %[[ALLOC_PONG_0:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
// CHECK: %[[ALLOC_PING_1:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
// CHECK: %[[ALLOC_PONG_1:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
// CHECK: %[[ALLOC_PING_2:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
// CHECK: %[[ALLOC_PONG_2:.*]] = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
//
// CHECK: %[[CMP0:.*]] = arith.cmpi slt, %[[C0]], %[[C1024]] : index
// CHECK: %[[ALLOC_CMP:.*]] = memref.alloc() : memref<i1>
// CHECK: memref.store %[[CMP0]], %[[ALLOC_CMP]][] : memref<i1>
// CHECK: %[[NON_ZERO_LOOP:.*]] = memref.load %[[ALLOC_CMP]][] : memref<i1>
//
// CHECK: scf.if %[[NON_ZERO_LOOP]] {
// CHECK:   %[[SUBVIEW_PROLOGUE_0:.*]] = memref.subview %[[ARG0]][%[[C0]], 0] [64, 2048] [1, 1]
// CHECK:   memref.copy %[[SUBVIEW_PROLOGUE_0]], %[[ALLOC_PING_0]]
// CHECK:   %[[SUBVIEW_PROLOGUE_1:.*]] = memref.subview %[[ARG1]][%[[C0]], 0] [64, 2048] [1, 1]
// CHECK:   memref.copy %[[SUBVIEW_PROLOGUE_1]], %[[ALLOC_PING_1]]
// CHECK:   %[[SUBVIEW_PROLOGUE_2:.*]] = memref.subview %[[ALLOC]][%[[C0]], 0] [64, 2048] [1, 1]
// CHECK:   memref.copy %[[SUBVIEW_PROLOGUE_2]], %[[ALLOC_PING_2]]
// CHECK: } {db_generic = 0 : i64, db_prologue}
//
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C1024]] step %[[C64]] {
// CHECK:   %[[IS_PING_STAGE:.*]] = memref.load %[[TOGGLE]][] : memref<i1>
// CHECK:   %[[NEXT_IV:.*]] = arith.addi %[[IV]], %[[C64]] : index
// CHECK:   %[[NEXT_NEXT_IV:.*]] = arith.addi %[[NEXT_IV]], %[[C64]] : index
// CHECK:   %[[IS_NOT_LAST_ITERATION:.*]] = arith.cmpi sle, %[[NEXT_NEXT_IV]], %[[C1024]] : index
// CHECK:   scf.if %[[IS_PING_STAGE]] {
// CHECK:     scf.if %[[IS_NOT_LAST_ITERATION]] {
// CHECK:       %[[SUBVIEW_PREFETCH_0:.*]] = memref.subview %[[ARG0]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_0]], %[[ALLOC_PONG_0]]
// CHECK:       %[[SUBVIEW_PREFETCH_1:.*]] = memref.subview %[[ARG1]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_1]], %[[ALLOC_PONG_1]]
// CHECK:       %[[SUBVIEW_PREFETCH_2:.*]] = memref.subview %[[ALLOC]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_2]], %[[ALLOC_PONG_2]]
// CHECK:     } {db_prefetch}
// CHECK:     scf.for %[[IV_I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:       scf.for %[[IV_J:.*]] = %[[C0]] to %[[C2048]] step %[[C32]] {
// CHECK:         %[[SUBVIEW_PING_0:.*]] = memref.subview %[[ALLOC_PING_0]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SUBVIEW_PING_1:.*]] = memref.subview %[[ALLOC_PING_1]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SUBVIEW_PING_2:.*]] = memref.subview %[[ALLOC_PING_2]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[VEC_READ_0:.*]] = vector.transfer_read %[[SUBVIEW_PING_0]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[VEC_READ_1:.*]] = vector.transfer_read %[[SUBVIEW_PING_1]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[SUBF:.*]] = arith.subf %[[VEC_READ_0]], %[[VEC_READ_1]] fastmath<fast> : vector<32xf32>
// CHECK:         %[[EXP:.*]] = math.exp %[[SUBF]] fastmath<fast> : vector<32xf32>
// CHECK:         vector.transfer_write %[[EXP]], %[[SUBVIEW_PING_2]][%[[C0]]] {in_bounds = [true]}
// CHECK-SAME:                             : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:       }
// CHECK:     }
// CHECK:     %[[SUBVIEW_WRITEBACK:.*]] = memref.subview %[[ALLOC]][%[[IV]], 0] [64, 2048] [1, 1]
// CHECK:     memref.copy %[[ALLOC_PING_2]], %[[SUBVIEW_WRITEBACK]]
// CHECK:     memref.store %[[FALSE]], %[[TOGGLE]][] : memref<i1>
// CHECK:   } {db_ping_kernel}
// CHECK:   %[[IS_PONG_STAGE:.*]] = arith.xori %[[IS_PING_STAGE]], %[[TRUE]] : i1
// CHECK:   scf.if %[[IS_PONG_STAGE]] {
// CHECK:     scf.if %[[IS_NOT_LAST_ITERATION]] {
// CHECK:       %[[SUBVIEW_PREFETCH_0:.*]] = memref.subview %[[ARG0]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_0]], %[[ALLOC_PING_0]]
// CHECK:       %[[SUBVIEW_PREFETCH_1:.*]] = memref.subview %[[ARG1]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_1]], %[[ALLOC_PING_1]]
// CHECK:       %[[SUBVIEW_PREFETCH_2:.*]] = memref.subview %[[ALLOC]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.copy %[[SUBVIEW_PREFETCH_2]], %[[ALLOC_PING_2]]
// CHECK:     } {db_prefetch}
// CHECK:     scf.for %[[IV_I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:       scf.for %[[IV_J:.*]] = %[[C0]] to %[[C2048]] step %[[C32]] {
// CHECK:         %[[SUBVIEW_PONG_0:.*]] = memref.subview %[[ALLOC_PONG_0]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SUBVIEW_PONG_1:.*]] = memref.subview %[[ALLOC_PONG_1]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SUBVIEW_PONG_2:.*]] = memref.subview %[[ALLOC_PONG_2]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[VEC_READ_0:.*]] = vector.transfer_read %[[SUBVIEW_PONG_0]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[VEC_READ_1:.*]] = vector.transfer_read %[[SUBVIEW_PONG_1]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[SUBF:.*]] = arith.subf %[[VEC_READ_0]], %[[VEC_READ_1]] fastmath<fast> : vector<32xf32>
// CHECK:         %[[EXP:.*]] = math.exp %[[SUBF]] fastmath<fast> : vector<32xf32>
// CHECK:         vector.transfer_write %[[EXP]], %[[SUBVIEW_PONG_2]][%[[C0]]] {in_bounds = [true]}
// CHECK-SAME:                            : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:       }
// CHECK:     }
// CHECK:     %[[SUBVIEW_WRITEBACK:.*]] = memref.subview %[[ALLOC]][%[[IV]], 0] [64, 2048] [1, 1]
// CHECK:     memref.copy %[[ALLOC_PONG_2]], %[[SUBVIEW_WRITEBACK]]
// CHECK:     memref.store %[[TRUE]], %[[TOGGLE]][] : memref<i1>
// CHECK:   } {db_pong_kernel}
// CHECK: } {db_generic = 0 : i64}
// CHECK: memref.copy %[[ALLOC]], %[[ARG2]]
// CHECK: return

module attributes {llvm.target_triple = "hexagon"} {
  func.func @kernel(%arg0: memref<1024x2048xf32>, %arg1: memref<1024x2048xf32>, %arg2: memref<1024x2048xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x2048xf32>
    scf.for %arg3 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview, %alloc_0 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_1 = memref.subview %arg1[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview_1, %alloc_2 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_3 = memref.subview %alloc[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview_3, %alloc_4 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      scf.for %arg4 = %c0 to %c64 step %c1 {
        scf.for %arg5 = %c0 to %c2048 step %c32 {
          %subview_5 = memref.subview %alloc_0[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %subview_6 = memref.subview %alloc_2[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %subview_7 = memref.subview %alloc_4[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %0 = vector.transfer_read %subview_5[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
          %1 = vector.transfer_read %subview_6[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
          %2 = arith.subf %0, %1 fastmath<fast> : vector<32xf32>
          %3 = math.exp %2 fastmath<fast> : vector<32xf32>
          vector.transfer_write %3, %subview_7[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
        }
      }
      memref.copy %alloc_4, %subview_3 : memref<64x2048xf32, 1> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
    } {all_parallel, tiled_generic}
    memref.copy %alloc, %arg2 : memref<1024x2048xf32> to memref<1024x2048xf32>
    return
  }
}
