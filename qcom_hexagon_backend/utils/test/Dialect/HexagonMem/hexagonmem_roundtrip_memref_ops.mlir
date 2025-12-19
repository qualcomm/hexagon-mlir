// RUN: linalg-hexagon-opt %s | linalg-hexagon-opt | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK: [[ALLOC0:%.*]] = hexagonmem.alloc() : memref<64xi8, 1>
// CHECK: [[ALLOC1:%.*]] = hexagonmem.alloc() : memref<64xi8, 1>
// CHECK: [[ALLOC2:%.*]] = hexagonmem.alloc() : memref<64xi8, 1>
// CHECK: hexagonmem.copy [[ARG0:%.*]], [[ALLOC0]] : memref<64xi8> to memref<64xi8, 1>
// CHECK: hexagonmem.copy [[ARG1:%.*]], [[ALLOC1]] : memref<64xi8> to memref<64xi8, 1>
// CHECK: linalg.add ins([[ALLOC0]], [[ALLOC1]] : memref<64xi8, 1>, memref<64xi8, 1>) outs([[ALLOC2]] : memref<64xi8, 1>)
// CHECK: hexagonmem.copy [[ALLOC2]], [[ARG2:%.*]] : memref<64xi8, 1> to memref<64xi8>
// CHECK: hexagonmem.dealloc [[ALLOC0]] : memref<64xi8, 1>
// CHECK: hexagonmem.dealloc [[ALLOC1]] : memref<64xi8, 1>
// CHECK: hexagonmem.dealloc [[ALLOC2]] : memref<64xi8, 1>

func.func @add(%arg0: memref<64xi8>, %arg1: memref<64xi8>, %arg2: memref<64xi8>) {
  %0 = hexagonmem.alloc() {alignment = 128 : i64} : memref<64xi8, 1>
  %1 = hexagonmem.alloc() {alignment = 128 : i64} : memref<64xi8, 1>
  %2 = hexagonmem.alloc() {alignment = 128 : i64} : memref<64xi8, 1>
  hexagonmem.copy %arg0, %0 : memref<64xi8> to memref<64xi8, 1>
  hexagonmem.copy %arg1, %1 : memref<64xi8> to memref<64xi8, 1>
  linalg.add ins(%0, %1 : memref<64xi8, 1>, memref<64xi8, 1>) outs(%2 : memref<64xi8, 1>)
  hexagonmem.copy %2, %arg2 : memref<64xi8, 1> to memref<64xi8>
  hexagonmem.dealloc %0 : memref<64xi8, 1>
  hexagonmem.dealloc %1 : memref<64xi8, 1>
  hexagonmem.dealloc %2 : memref<64xi8, 1>
  return
}
