// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s
// This test checks that we can run linalg-to-llvm pass pipeline on a module
// with multiple functions
module {
  func.func private @external_func(memref<32xf32>, memref<32xf32>) -> ()
  func.func @copy(%arg0: memref<32xf32>, %arg1: memref<32xf32>) {
    memref.copy %arg0, %arg1 : memref<32xf32> to memref<32xf32>
    return
  }

  func.func @copy_wrapper(%arg0: memref<32xf32>, %arg1: memref<32xf32>) {
    call @external_func(%arg0, %arg1) : (memref<32xf32>, memref<32xf32>) -> ()
    return
  }
}

// CHECK-LABEL: @copy
// CHECK-LABEL: @copy_wrapper
