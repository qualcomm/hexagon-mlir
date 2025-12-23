  // RUN: linalg-hexagon-opt %s -dma-to-llvm | FileCheck %s
  // CHECK:       llvm.func @hexagon_runtime_dma_wait(i32)
  // CHECK:       llvm.func @hexagon_runtime_dma_start(!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK:       llvm.func @hexagon_runtime_dma2d_start(!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK-LABEL: func.func @dma_copy
  // CHECK:       %[[VAL_X:.*]] = llvm.call @hexagon_runtime_dma_start
  // CHECK:       llvm.call @hexagon_runtime_dma_wait


func.func @dma_copy(%arg0: memref<1024x1024xi8, strided<[1024, 1]>>, %arg1: memref<1024x1024xi8, strided<[1024, 1]>>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %c1M = arith.constant 1048576 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %c1M, %tag[%c0] : memref<1024x1024xi8, strided<[1024, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %c1M : memref<1xi32, strided<[1]>>
  return
}


  // CHECK-LABEL: func.func @dma_2d_strided_copy
  // CHECK:       %[[VAL_X:.*]] = llvm.call @hexagon_runtime_dma2d_start
  // CHECK:       llvm.call @hexagon_runtime_dma_wait
  // CHECK:       %[[VAL_Y:.*]] = llvm.call @hexagon_runtime_dma2d_start
  // CHECK:       llvm.call @hexagon_runtime_dma_wait


func.func @dma_2d_strided_copy(%arg0: memref<1024x1024xi8, strided<[2048, 1]>>, %arg1: memref<1024x1024xi8, strided<[1024, 1]>, 1>, %arg2: memref<1024x1024xf32, strided<[2048, 1]>>, %arg3: memref<1024x1024xf32, strided<[1024, 1]>, 1>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %number_elements = arith.constant 1048576 : index // 1024 * 1024
  %cStride = arith.constant 2048 : index
  %cWidth = arith.constant 1024 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %number_elements, %tag[%c0], %cStride, %cWidth : memref<1024x1024xi8, strided<[2048, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>, 1>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %number_elements : memref<1xi32, strided<[1]>>
  memref.dma_start %arg2[%c0, %c0], %arg3[%c0, %c0], %number_elements, %tag[%c0], %cStride, %cWidth : memref<1024x1024xf32, strided<[2048, 1]>>, memref<1024x1024xf32, strided<[1024, 1]>, 1>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %number_elements : memref<1xi32, strided<[1]>>
  return
}


