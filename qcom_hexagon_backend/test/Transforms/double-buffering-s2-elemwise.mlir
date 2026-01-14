// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-double-buffer-generic-s2),cse,canonicalize)' %s | FileCheck %s

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x2048xf32>, %[[ARG1:.*]]: memref<1024x2048xf32>, %[[ARG2:.*]]: memref<1024x2048xf32>)
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C131072:.*]] = arith.constant 131072 : index
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1024x2048xf32>
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
// CHECK: %[[ALLOC_CMP:.*]] = memref.alloc() : memref<i1>
// CHECK: memref.store %[[TRUE]], %[[ALLOC_CMP]][] : memref<i1>
// CHECK: %[[NON_ZERO_LOOP:.*]] = memref.load %[[ALLOC_CMP]][] : memref<i1>
//
// CHECK: %[[PING_TAG_0:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[PONG_TAG_0:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[PING_TAG_1:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[PONG_TAG_1:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[PING_TAG_2:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[PONG_TAG_2:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[WRITEBACK_PING_TAG:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[WRITEBACK_PONG_TAG:.*]] = memref.alloc() : memref<1xi32>
//
// CHECK: scf.if %[[NON_ZERO_LOOP]] {
// CHECK:   %[[SV_PROLOGUE_0:.*]] = memref.subview %[[ARG0]][0, 0] [64, 2048] [1, 1]
// CHECK:   memref.dma_start %[[SV_PROLOGUE_0]][%[[C0]], %[[C0]]], %[[ALLOC_PING_0]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_0]][%[[C0]]]
// CHECK:   %[[SV_PROLOGUE_1:.*]] = memref.subview %[[ARG1]][0, 0] [64, 2048] [1, 1]
// CHECK:   memref.dma_start %[[SV_PROLOGUE_1]][%[[C0]], %[[C0]]], %[[ALLOC_PING_1]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_1]][%[[C0]]]
// CHECK:   %[[SV_PROLOGUE_2:.*]] = memref.subview %[[ALLOC]][0, 0] [64, 2048] [1, 1]
// CHECK:   memref.dma_start %[[SV_PROLOGUE_2]][%[[C0]], %[[C0]]], %[[ALLOC_PING_2]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_2]][%[[C0]]]
// CHECK: } {db_generic = 0 : i64, db_prologue}
//
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C1024]] step %[[C64]] {
// CHECK:   %[[IS_PING_STAGE:.*]] = memref.load %[[TOGGLE]][] : memref<i1>
// CHECK:   %[[NEXT_IV:.*]] = arith.addi %[[IV]], %[[C64]] : index
// CHECK:   %[[NEXT_NEXT_IV:.*]] = arith.addi %[[IV]], %[[C128]] : index
// CHECK:   %[[IS_NOT_LAST_ITERATION:.*]] = arith.cmpi sle, %[[NEXT_NEXT_IV]], %[[C1024]] : index
// CHECK:   scf.if %[[IS_PING_STAGE]] {
// CHECK:     scf.if %[[IS_NOT_LAST_ITERATION]] {
// CHECK:       %[[SV_PREFETCH_0:.*]] = memref.subview %[[ARG0]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_0]][%[[C0]], %[[C0]]], %[[ALLOC_PONG_0]][%[[C0]], %[[C0]]], %[[C131072]], %[[PONG_TAG_0]][%[[C0]]]
// CHECK:       %[[SV_PREFETCH_1:.*]] = memref.subview %[[ARG1]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_1]][%[[C0]], %[[C0]]], %[[ALLOC_PONG_1]][%[[C0]], %[[C0]]], %[[C131072]], %[[PONG_TAG_1]][%[[C0]]]
// CHECK:       %[[SV_PREFETCH_2:.*]] = memref.subview %[[ALLOC]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_2]][%[[C0]], %[[C0]]], %[[ALLOC_PONG_2]][%[[C0]], %[[C0]]], %[[C131072]], %[[PONG_TAG_2]][%[[C0]]]
// CHECK:     } {db_prefetch}
// CHECK:     memref.dma_wait %[[PING_TAG_0]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.dma_wait %[[PING_TAG_1]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.dma_wait %[[PING_TAG_2]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     scf.for %[[IV_I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:       scf.for %[[IV_J:.*]] = %[[C0]] to %[[C2048]] step %[[C32]] {
// CHECK:         %[[SV_PING_0:.*]] = memref.subview %[[ALLOC_PING_0]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SV_PING_1:.*]] = memref.subview %[[ALLOC_PING_1]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SV_PING_2:.*]] = memref.subview %[[ALLOC_PING_2]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[VEC_READ_0:.*]] = vector.transfer_read %[[SV_PING_0]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[VEC_READ_1:.*]] = vector.transfer_read %[[SV_PING_1]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[SUBF:.*]] = arith.subf %[[VEC_READ_0]], %[[VEC_READ_1]] fastmath<fast> : vector<32xf32>
// CHECK:         %[[EXP:.*]] = math.exp %[[SUBF]] fastmath<fast> : vector<32xf32>
// CHECK:         vector.transfer_write %[[EXP]], %[[SV_PING_2]][%[[C0]]] {in_bounds = [true]}
// CHECK-SAME:                             : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:       }
// CHECK:     }
// CHECK:     %[[SV_WRITEBACK:.*]] = memref.subview %[[ALLOC]][%[[IV]], 0] [64, 2048] [1, 1]
// CHECK:     memref.dma_start %[[ALLOC_PING_2]][%[[C0]], %[[C0]]], %[[SV_WRITEBACK]][%[[C0]], %[[C0]]], %[[C131072]], %[[WRITEBACK_PING_TAG]][%[[C0]]]
// CHECK:     memref.dma_wait %[[WRITEBACK_PING_TAG]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.store %[[FALSE]], %[[TOGGLE]][] : memref<i1>
// CHECK:   } {db_ping_kernel}
// CHECK:   %[[IS_PONG_STAGE:.*]] = arith.xori %[[IS_PING_STAGE]], %[[TRUE]] : i1
// CHECK:   scf.if %[[IS_PONG_STAGE]] {
// CHECK:     scf.if %[[IS_NOT_LAST_ITERATION]] {
// CHECK:       %[[SV_PREFETCH_0:.*]] = memref.subview %[[ARG0]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_0]][%[[C0]], %[[C0]]], %[[ALLOC_PING_0]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_0]][%[[C0]]]
// CHECK:       %[[SV_PREFETCH_1:.*]] = memref.subview %[[ARG1]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_1]][%[[C0]], %[[C0]]], %[[ALLOC_PING_1]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_1]][%[[C0]]]
// CHECK:       %[[SV_PREFETCH_2:.*]] = memref.subview %[[ALLOC]][%[[NEXT_IV]], 0] [64, 2048] [1, 1]
// CHECK:       memref.dma_start %[[SV_PREFETCH_2]][%[[C0]], %[[C0]]], %[[ALLOC_PING_2]][%[[C0]], %[[C0]]], %[[C131072]], %[[PING_TAG_2]][%[[C0]]]
// CHECK:     } {db_prefetch}
// CHECK:     memref.dma_wait %[[PONG_TAG_0]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.dma_wait %[[PONG_TAG_1]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.dma_wait %[[PONG_TAG_2]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     scf.for %[[IV_I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:       scf.for %[[IV_J:.*]] = %[[C0]] to %[[C2048]] step %[[C32]] {
// CHECK:         %[[SV_PONG_0:.*]] = memref.subview %[[ALLOC_PONG_0]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SV_PONG_1:.*]] = memref.subview %[[ALLOC_PONG_1]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[SV_PONG_2:.*]] = memref.subview %[[ALLOC_PONG_2]][%[[IV_I]], %[[IV_J]]] [1, 32] [1, 1]
// CHECK:         %[[VEC_READ_0:.*]] = vector.transfer_read %[[SV_PONG_0]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[VEC_READ_1:.*]] = vector.transfer_read %[[SV_PONG_1]][%[[C0]]], %[[CST]] {in_bounds = [true]}
// CHECK-SAME:                             : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
// CHECK:         %[[SUBF:.*]] = arith.subf %[[VEC_READ_0]], %[[VEC_READ_1]] fastmath<fast> : vector<32xf32>
// CHECK:         %[[EXP:.*]] = math.exp %[[SUBF]] fastmath<fast> : vector<32xf32>
// CHECK:         vector.transfer_write %[[EXP]], %[[SV_PONG_2]][%[[C0]]] {in_bounds = [true]}
// CHECK-SAME:                            : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
// CHECK:       }
// CHECK:     }
// CHECK:     %[[SV_WRITEBACK:.*]] = memref.subview %[[ALLOC]][%[[IV]], 0] [64, 2048] [1, 1]
// CHECK:     memref.dma_start %[[ALLOC_PONG_2]][%[[C0]], %[[C0]]], %[[SV_WRITEBACK]][%[[C0]], %[[C0]]], %[[C131072]], %[[WRITEBACK_PONG_TAG]][%[[C0]]]
// CHECK:     memref.dma_wait %[[WRITEBACK_PONG_TAG]][%[[C0]]], %[[C131072]] : memref<1xi32>
// CHECK:     memref.store %[[TRUE]], %[[TOGGLE]][] : memref<i1>
// CHECK:   } {db_pong_kernel}
// CHECK: } {db_generic = 0 : i64}
// CHECK: memref.dealloc %[[PING_TAG_0]] : memref<1xi32>
// CHECK: memref.dealloc %[[PONG_TAG_0]] : memref<1xi32>
// CHECK: memref.dealloc %[[PING_TAG_1]] : memref<1xi32>
// CHECK: memref.dealloc %[[PONG_TAG_1]] : memref<1xi32>
// CHECK: memref.dealloc %[[PING_TAG_2]] : memref<1xi32>
// CHECK: memref.dealloc %[[PONG_TAG_2]] : memref<1xi32>
// CHECK: memref.dealloc %[[WRITEBACK_PING_TAG]] : memref<1xi32>
// CHECK: memref.dealloc %[[WRITEBACK_PONG_TAG]] : memref<1xi32>
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
    %true = arith.constant true
    %false = arith.constant false
    %alloc_0 = memref.alloc() : memref<i1>
    memref.store %true, %alloc_0[] : memref<i1>
    %alloc_1 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %alloc_2 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %alloc_3 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %alloc_4 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %alloc_5 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %alloc_6 = memref.alloc() {alignment = 2048 : i64} : memref<64x2048xf32, 1>
    %0 = arith.cmpi slt, %c0, %c1024 : index
    %alloc_7 = memref.alloc() : memref<i1>
    memref.store %0, %alloc_7[] : memref<i1>
    %1 = memref.load %alloc_7[] : memref<i1>
    scf.if %1 {
      %subview = memref.subview %arg0[%c0, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      memref.copy %subview, %alloc_1 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_8 = memref.subview %arg1[%c0, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      memref.copy %subview_8, %alloc_3 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_9 = memref.subview %alloc[%c0, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      memref.copy %subview_9, %alloc_5 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
    } {db_generic = 0 : i64, db_prologue}
    scf.for %arg3 = %c0 to %c1024 step %c64 {
      %2 = memref.load %alloc_0[] : memref<i1>
      %3 = arith.addi %arg3, %c64 : index
      %4 = arith.addi %3, %c64 : index
      %5 = arith.cmpi sle, %4, %c1024 : index
      scf.if %2 {
        scf.if %5 {
          %subview_8 = memref.subview %arg0[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_8, %alloc_2 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
          %subview_9 = memref.subview %arg1[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_9, %alloc_4 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
          %subview_10 = memref.subview %alloc[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_10, %alloc_6 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
        } {db_prefetch}
        scf.for %arg4 = %c0 to %c64 step %c1 {
          scf.for %arg5 = %c0 to %c2048 step %c32 {
            %subview_8 = memref.subview %alloc_1[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %subview_9 = memref.subview %alloc_3[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %subview_10 = memref.subview %alloc_5[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %7 = vector.transfer_read %subview_8[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
            %8 = vector.transfer_read %subview_9[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
            %9 = arith.subf %7, %8 fastmath<fast> : vector<32xf32>
            %10 = math.exp %9 fastmath<fast> : vector<32xf32>
            vector.transfer_write %10, %subview_10[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
          }
        }
        %subview = memref.subview %alloc[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
        memref.copy %alloc_5, %subview : memref<64x2048xf32, 1> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
        memref.store %false, %alloc_0[] : memref<i1>
      } {db_ping_kernel}
      %6 = arith.xori %2, %true : i1
      scf.if %6 {
        scf.if %5 {
          %subview_8 = memref.subview %arg0[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_8, %alloc_1 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
          %subview_9 = memref.subview %arg1[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_9, %alloc_3 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
          %subview_10 = memref.subview %alloc[%3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
          memref.copy %subview_10, %alloc_5 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
        } {db_prefetch}
        scf.for %arg4 = %c0 to %c64 step %c1 {
          scf.for %arg5 = %c0 to %c2048 step %c32 {
            %subview_8 = memref.subview %alloc_2[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %subview_9 = memref.subview %alloc_4[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %subview_10 = memref.subview %alloc_6[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
            %7 = vector.transfer_read %subview_8[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
            %8 = vector.transfer_read %subview_9[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
            %9 = arith.subf %7, %8 fastmath<fast> : vector<32xf32>
            %10 = math.exp %9 fastmath<fast> : vector<32xf32>
            vector.transfer_write %10, %subview_10[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
          }
        }
        %subview = memref.subview %alloc[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
        memref.copy %alloc_6, %subview : memref<64x2048xf32, 1> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
        memref.store %true, %alloc_0[] : memref<i1>
      } {db_pong_kernel}
    } {db_generic = 0 : i64}
    memref.copy %alloc, %arg2 : memref<1024x2048xf32> to memref<1024x2048xf32>
    return
  }
}
