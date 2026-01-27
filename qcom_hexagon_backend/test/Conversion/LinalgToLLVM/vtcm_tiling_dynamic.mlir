// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(linalg-generalize-named-ops,\
// RUN:   func.func(vtcm-tiling), one-shot-bufferize,\
// RUN:   func.func(buffer-loop-hoisting), canonicalize, buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s -check-prefixes=CHECK

module {
  func.func @dynamic_sizes(%arg0: memref<1024x?xf32>, %arg1: memref<?x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<1024x?xf32> to tensor<1024x?xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<?x1024xf32> to tensor<?x1024xf32>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<1024x1024xf32>  to tensor<1024x1024xf32>

    %3 = linalg.matmul ins(%0, %1 : tensor<1024x?xf32>, tensor<?x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }
}

// CHECK-LABEL: @dynamic_sizes
// CHECK-SAME: %[[X:.+]]: memref<1024x?xf32>, %[[Y:.+]]: memref<?x1024xf32>, %[[Z:.+]]: memref<1024x1024xf32>
//
// CHECK-DAG: %[[DIM_K:.+]] = memref.dim %arg0, %c1 : memref<1024x?xf32>
// CHECK-DAG: %[[TZ:.+]] = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32, 1>
// CHECK:     scf.for %[[I:.+]] = %c0 to %c1024 step %c256
// CHECK-NEXT:   scf.for %[[J:.+]] = %c0 to %c1024 step %c512
// CHECK-NEXT:     scf.for %[[K:.+]] = %c0 to %[[DIM_K]] step %c256
// CHECK:            %[[THIS_K:.+]] = affine.min #map(%[[K]])[%[[DIM_K]]]
//
// CHECK:            %[[ViewX:.+]] = memref.subview %[[X]][%[[I]], %[[K]]] [256, %[[THIS_K]]] [1, 1] : memref<1024x?xf32> to memref<256x?xf32, strided<[?, 1], offset: ?>>
// CHECK:            %[[TX:.+]] = memref.alloc(%[[THIS_K]]) {alignment = 64 : i64} : memref<256x?xf32, 1>
// CHECK:            memref.copy %[[ViewX]], %[[TX]] : memref<256x?xf32, strided<[?, 1], offset: ?>> to memref<256x?xf32, 1>
//
// CHECK:            %[[ViewY:.+]] = memref.subview %[[Y]][%[[K]], %[[J]]] [%[[THIS_K]], 512] [1, 1] : memref<?x1024xf32> to memref<?x512xf32, strided<[1024, 1], offset: ?>>
// CHECK:            %[[TY:.+]] = memref.alloc(%[[THIS_K]]) {alignment = 64 : i64} : memref<?x512xf32, 1>
// CHECK:            memref.copy %[[ViewY]], %[[TY]] : memref<?x512xf32, strided<[1024, 1], offset: ?>> to memref<?x512xf32, 1>
//
// CHECK:            %[[ViewZ:.+]] = memref.subview %[[Z]][%[[I]], %[[J]]] [256, 512] [1, 1] : memref<1024x1024xf32> to memref<256x512xf32, strided<[1024, 1], offset: ?>>
// CHECK:            memref.copy %[[ViewZ]], %[[TZ]] :  memref<256x512xf32, strided<[1024, 1], offset: ?>> to memref<256x512xf32, 1>
// CHECK:            linalg.generic {{.*}} ins(%[[TX]], %[[TY]] : memref<256x?xf32, 1>, memref<?x512xf32, 1>) outs(%[[TZ]] : memref<256x512xf32, 1>)
//
// CHECK:            memref.copy %[[TZ]], %[[ViewZ]] : memref<256x512xf32, 1> to memref<256x512xf32, strided<[1024, 1], offset: ?>>
// CHECK:            memref.dealloc %[[TX]] : memref<256x?xf32, 1>
// CHECK:            memref.dealloc %[[TY]] : memref<?x512xf32, 1>
// CHECK:            memref.dealloc %[[TZ]] : memref<256x512xf32, 1>
