// RUN: linalg-hexagon-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func.func @empty_tensor_dynamic_dim(%arg0 : index) -> (index) {
  %c2 = arith.constant 2 : index
  %0 = tensor.empty(%arg0) : tensor<4x5x?xf32>
  %1 = tensor.dim %0, %c2 : tensor<4x5x?xf32>
  return %1 : index
}
// CHECK: func @empty_tensor_dynamic_dim
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK: return %[[ARG0]]
