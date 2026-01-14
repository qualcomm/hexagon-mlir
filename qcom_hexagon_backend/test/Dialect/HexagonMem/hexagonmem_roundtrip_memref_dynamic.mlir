// RUN: linalg-hexagon-opt %s | linalg-hexagon-opt | FileCheck %s

// CHECK-LABEL:   func.func @add
// CHECK:           %[[VAL_4:.*]] = hexagonmem.alloc(%[[VAL_3:.*]]) {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           %[[VAL_5:.*]] = hexagonmem.alloc(%[[VAL_3:.*]]) {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           %[[VAL_6:.*]] = hexagonmem.alloc(%[[VAL_3:.*]]) {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           hexagonmem.copy %[[VAL_0:.*]], %[[VAL_4]] : memref<?xi8> to memref<?xi8, 1>
// CHECK:           hexagonmem.copy %[[VAL_1:.*]], %[[VAL_5]] : memref<?xi8> to memref<?xi8, 1>
// CHECK:           linalg.add ins(%[[VAL_4]], %[[VAL_5]] : memref<?xi8, 1>, memref<?xi8, 1>) outs(%[[VAL_6]] : memref<?xi8, 1>)
// CHECK:           hexagonmem.copy %[[VAL_6]], %[[VAL_2:.*]] : memref<?xi8, 1> to memref<?xi8>
// CHECK:           hexagonmem.dealloc %[[VAL_4]] {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           hexagonmem.dealloc %[[VAL_5]] {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           hexagonmem.dealloc %[[VAL_6]] {bufferization.manual_deallocation} : memref<?xi8, 1>
// CHECK:           return
// CHECK:         }

func.func @add(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %n: index) {
  %0 = hexagonmem.alloc(%n) {bufferization.manual_deallocation, alignment = 128 : i64} : memref<?xi8, 1>
  %1 = hexagonmem.alloc(%n) {bufferization.manual_deallocation, alignment = 128 : i64} : memref<?xi8, 1>
  %2 = hexagonmem.alloc(%n) {bufferization.manual_deallocation, alignment = 128 : i64} : memref<?xi8, 1>
  hexagonmem.copy %arg0, %0 : memref<?xi8> to memref<?xi8, 1>
  hexagonmem.copy %arg1, %1 : memref<?xi8> to memref<?xi8, 1>
  linalg.add ins(%0, %1 : memref<?xi8, 1>, memref<?xi8, 1>) outs(%2 : memref<?xi8, 1>)
  hexagonmem.copy %2, %arg2 : memref<?xi8, 1> to memref<?xi8>
  hexagonmem.dealloc %0 {bufferization.manual_deallocation} : memref<?xi8, 1>
  hexagonmem.dealloc %1 {bufferization.manual_deallocation} : memref<?xi8, 1>
  hexagonmem.dealloc %2 {bufferization.manual_deallocation} : memref<?xi8, 1>
  return
}
