// RUN: linalg-hexagon-opt %s --pass-pipeline="builtin.module(func.func(fold-casts-into-matmul))" -split-input-file | FileCheck %s

// ============================================================================
// Test 1: Fold both LHS and RHS FP16→FP32 casts into matmul
// ============================================================================

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fold_both_casts
func.func @fold_both_casts(%lhs_fp16: tensor<32x64xf16>, %rhs_fp16: tensor<64x128xf16>) -> tensor<32x128xf32> {
  %empty_lhs = tensor.empty() : tensor<32x64xf32>
  %empty_rhs = tensor.empty() : tensor<64x128xf32>
  %empty_out = tensor.empty() : tensor<32x128xf32>

  // Cast LHS from FP16 to FP32
  // CHECK-NOT: linalg.generic
  %lhs_fp32 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs_fp16 : tensor<32x64xf16>) outs(%empty_lhs : tensor<32x64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %cast = arith.extf %in : f16 to f32
    linalg.yield %cast : f32
  } -> tensor<32x64xf32>

  // Cast RHS from FP16 to FP32
  // CHECK-NOT: linalg.generic
  %rhs_fp32 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%rhs_fp16 : tensor<64x128xf16>) outs(%empty_rhs : tensor<64x128xf32>) {
  ^bb0(%in: f16, %out: f32):
    %cast = arith.extf %in : f16 to f32
    linalg.yield %cast : f32
  } -> tensor<64x128xf32>

  // Matmul with FP32 inputs and output
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf16>, tensor<64x128xf16>)
  // CHECK-SAME: outs(%{{.*}} : tensor<32x128xf32>)
  %result = linalg.matmul ins(%lhs_fp32, %rhs_fp32 : tensor<32x64xf32>, tensor<64x128xf32>)
                          outs(%empty_out : tensor<32x128xf32>) -> tensor<32x128xf32>

  return %result : tensor<32x128xf32>
}

// -----

// ============================================================================
// Test 2: Fold only LHS cast (RHS is already FP32)
// ============================================================================

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fold_lhs_only
func.func @fold_lhs_only(%lhs_fp16: tensor<32x64xf16>, %rhs_fp32: tensor<64x128xf32>) -> tensor<32x128xf32> {
  %empty_lhs = tensor.empty() : tensor<32x64xf32>
  %empty_out = tensor.empty() : tensor<32x128xf32>

  // Cast LHS from FP16 to FP32
  // CHECK-NOT: linalg.generic
  %lhs_fp32 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs_fp16 : tensor<32x64xf16>) outs(%empty_lhs : tensor<32x64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %cast = arith.extf %in : f16 to f32
    linalg.yield %cast : f32
  } -> tensor<32x64xf32>

  // Matmul with mixed inputs: FP16 LHS, FP32 RHS
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf16>, tensor<64x128xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<32x128xf32>)
  %result = linalg.matmul ins(%lhs_fp32, %rhs_fp32 : tensor<32x64xf32>, tensor<64x128xf32>)
                          outs(%empty_out : tensor<32x128xf32>) -> tensor<32x128xf32>

  return %result : tensor<32x128xf32>
}

// -----

// ============================================================================
// Test 3: No folding - matmul with FP32 output but no cast producers
// ============================================================================

// CHECK-LABEL: func.func @no_fold_no_casts
func.func @no_fold_no_casts(%lhs: tensor<32x64xf32>, %rhs: tensor<64x128xf32>) -> tensor<32x128xf32> {
  %empty_out = tensor.empty() : tensor<32x128xf32>

  // No casts to fold - should remain unchanged
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32>)
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x64xf32>, tensor<64x128xf32>)
                          outs(%empty_out : tensor<32x128xf32>) -> tensor<32x128xf32>

  return %result : tensor<32x128xf32>
}
