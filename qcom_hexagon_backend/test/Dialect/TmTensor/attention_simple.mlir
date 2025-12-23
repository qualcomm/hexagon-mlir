// RUN: linalg-hexagon-opt %s | FileCheck %s

// Test that tm_tensor.attention operation passes verification and is printed correctly
func.func @attention_simple(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>,
                            %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x4x4xf32>) -> tensor<2x4x8xf32> {
  %0 = tensor.empty() : tensor<2x4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  // CHECK: tm_tensor.attention
  // CHECK-SAME: ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x4xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  %2 = tm_tensor.attention
         ins(%arg0, %arg1, %arg2, %arg3
             : tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x4xf32>)
         outs(%1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %2 : tensor<2x4x8xf32>
}
