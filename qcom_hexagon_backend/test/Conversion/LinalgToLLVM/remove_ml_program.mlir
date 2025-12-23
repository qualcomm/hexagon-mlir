// RUN: linalg-hexagon-opt %s -remove-ml-program | FileCheck %s

// CHECK-NOT: ml_program.global private mutable @global_seed

module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    return %sum : f32
  }
}
