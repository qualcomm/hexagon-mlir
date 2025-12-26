// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s

module {
  func.func @buffer_alloc(%arg0: memref<32xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    memref.copy %alloc, %arg0 : memref<32xf32> to memref<32xf32>
    return
  }
}

// CHECK-LABEL: @buffer_alloc
// CHECK:      %[[ALLOC:.+]] = llvm.call @malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:   llvm.call @free(%[[ALLOC]]) : (!llvm.ptr) -> ()
