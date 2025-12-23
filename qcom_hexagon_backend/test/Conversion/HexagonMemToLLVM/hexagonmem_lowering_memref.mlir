// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @hexagon_runtime_free_1d(!llvm.ptr)
// CHECK:         llvm.func @hexagon_runtime_copy(!llvm.ptr, !llvm.ptr, i32, i1, i1)
// CHECK:         llvm.func @hexagon_runtime_alloc_1d(i32, i64, i1) -> !llvm.ptr
// CHECK-LABEL:   func.func @add
// CHECK: [[ALLOC3:%.*]] = llvm.call @hexagon_runtime_alloc_1d
// CHECK: [[CAST2:%.*]] = builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<64xi8, 1>
// CHECK: [[ALLOC4:%.*]] = memref.alloc() : memref<64xi8>
// CHECK: [[ALLOC5:%.*]] = llvm.call @hexagon_runtime_alloc_1d
// CHECK: [[CAST3:%.*]] = builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<64xi8, 1>
// CHECK: llvm.call @hexagon_runtime_copy
// CHECK: memref.copy %arg1, [[ALLOC4]] : memref<64xi8> to memref<64xi8>
// CHECK: linalg.add ins([[CAST2]], [[ALLOC4]] : memref<64xi8, 1>, memref<64xi8>) outs([[CAST3]] : memref<64xi8, 1>)
// CHECK: llvm.call @hexagon_runtime_copy
// CHECK: llvm.call @hexagon_runtime_free_1d
// CHECK: memref.dealloc [[ALLOC4]] : memref<64xi8>
// CHECK: llvm.call @hexagon_runtime_free_1d
// CHECK: return

func.func @add(%arg0: memref<64xi8>, %arg1: memref<64xi8>, %arg2: memref<64xi8>) {
  %0 = hexagonmem.alloc() {bufferization.manual_deallocation, alignment = 128 : i64} : memref<64xi8, 1>
  %1 = hexagonmem.alloc() {bufferization.manual_deallocation, alignment = 128 : i64} : memref<64xi8>
  %2 = hexagonmem.alloc() {bufferization.manual_deallocation, alignment = 128 : i64} : memref<64xi8, 1>
  hexagonmem.copy %arg0, %0 : memref<64xi8> to memref<64xi8, 1>
  hexagonmem.copy %arg1, %1 : memref<64xi8> to memref<64xi8>
  linalg.add ins(%0, %1 : memref<64xi8, 1>, memref<64xi8>) outs(%2 : memref<64xi8, 1>)
  hexagonmem.copy %2, %arg2 : memref<64xi8, 1> to memref<64xi8>
  hexagonmem.dealloc %0 {bufferization.manual_deallocation} : memref<64xi8, 1>
  hexagonmem.dealloc %1 {bufferization.manual_deallocation} : memref<64xi8>
  hexagonmem.dealloc %2 {bufferization.manual_deallocation} : memref<64xi8, 1>
  return
}
