// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor))' %s | FileCheck %s

func.func @SDPA(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>,
                            %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x4x4xf32>) -> tensor<2x4x8xf32> {
  %0 = tensor.empty() : tensor<2x4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  %2 = tm_tensor.attention
         ins(%arg0, %arg1, %arg2, %arg3
             : tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x4xf32>)
         outs(%1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %2 : tensor<2x4x8xf32>
}

// CHECK-LABEL: func.func @SDPA
// CHECK: %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.353553385 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 0.000000e+00 : f32

// CHECK: %[[TRANSPOSED:.*]] = linalg.transpose ins(%arg1 : tensor<2x4x8xf32>) outs(%{{.*}} : tensor<2x8x4xf32>)
// CHECK-SAME:                  permutation = [0, 2, 1]

// CHECK: %[[QK:.*]] = linalg.batch_matmul ins(%arg0, %[[TRANSPOSED]] : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[SCALED:.*]] = linalg.generic {{.*}} ins(%[[QK]] : tensor<2x4x4xf32>) outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.mulf %[[IN]], %[[CST_0]] : f32

// CHECK: %[[QK_BIAS:.*]] = linalg.add ins(%[[SCALED]], %arg3 : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[QK_BIAS]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK-NEXT: %{{.*}} = arith.maximumf %[[IN]], %[[INIT]] : f32

// CHECK: %[[SUBBED:.*]] = linalg.generic {{.*}} ins(%[[QK_BIAS]], %[[REDUCED]] : tensor<2x4x4xf32>, tensor<2x4xf32>)
// CHECK-SAME:                  outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN_3:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.subf %[[IN]], %[[IN_3]] : f32

// CHECK: %[[EXP:.*]] = linalg.exp ins(%[[SUBBED]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[REDUCED_2:.*]] = linalg.reduce ins(%[[EXP]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK-NEXT: %{{.*}} = arith.addf %[[IN]], %[[INIT]] : f32

// CHECK: %[[SOFTMAX:.*]] = linalg.generic {{.*}} ins(%[[EXP]], %[[REDUCED_2]] : tensor<2x4x4xf32>, tensor<2x4xf32>)
// CHECK-SAME:                  outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN_3:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.divf %[[IN]], %[[IN_3]] : f32

// CHECK: %[[RESULT:.*]] = linalg.batch_matmul ins(%[[SOFTMAX]], %arg2 : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
// CHECK-SAME:                outs(%{{.*}} : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK: return %[[RESULT]] : tensor<2x4x8xf32>
