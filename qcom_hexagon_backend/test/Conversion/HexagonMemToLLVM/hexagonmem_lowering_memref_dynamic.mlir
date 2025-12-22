// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @hexagon_runtime_free_1d(!llvm.ptr)
// CHECK:         llvm.func @hexagon_runtime_copy(!llvm.ptr, !llvm.ptr, i32, i1, i1)
// CHECK:         llvm.func @hexagon_runtime_alloc_1d(i32, i64, i1) -> !llvm.ptr

// CHECK: %[[ALLOC1:.*]] = llvm.call @hexagon_runtime_alloc_1d({{.*}}) : (i32, i64, i1) -> !llvm.ptr
// CHECK: %[[ALLOC2:.*]] = llvm.call @hexagon_runtime_alloc_1d({{.*}}) : (i32, i64, i1) -> !llvm.ptr
// CHECK: %[[ALLOC3:.*]] = llvm.call @hexagon_runtime_alloc_1d({{.*}}) : (i32, i64, i1) -> !llvm.ptr
// CHECK: llvm.call @hexagon_runtime_copy(%[[DST1:.*]], %[[SRC1:.*]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK: llvm.call @hexagon_runtime_copy(%[[DST2:.*]], %[[SRC2:.*]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK: linalg.add
// CHECK: llvm.call @hexagon_runtime_copy(%[[DST3:.*]], %[[SRC3:.*]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK: llvm.call @hexagon_runtime_free_1d({{.*}}) : (!llvm.ptr) -> ()
// CHECK: llvm.call @hexagon_runtime_free_1d({{.*}}) : (!llvm.ptr) -> ()
// CHECK: llvm.call @hexagon_runtime_free_1d({{.*}}) : (!llvm.ptr) -> ()


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
