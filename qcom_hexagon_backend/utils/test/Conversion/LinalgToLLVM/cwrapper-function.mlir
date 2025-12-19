// RUN: linalg-hexagon-opt %s -linalg-to-llvm="enable-multi-threading=true" | FileCheck %s

// This test checks that even when multiple functions are present in the final MLIR,
// the "_mlir_ciface_" functions are only generated for the functions in the original MLIR.
module {
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    return %sum : f32
  }
}

// CHECK: llvm.func @add
// CHECK: llvm.func @_mlir_ciface_add
// CHECK: llvm.func @mlirAsyncRuntimeAddRef
// CHECK-NOT: llvm.func @_mlir_ciface_mlirAsyncRuntimeAddRef
