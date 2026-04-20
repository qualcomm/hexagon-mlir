// RUN: linalg-hexagon-opt %s --pass-pipeline="builtin.module(func.func(matmul-to-hexkl{mode=macro}))" -split-input-file | FileCheck %s

// ============================================================================
// Test 1: Basic f16 matmul with constant weights (should convert)
// ============================================================================

// CHECK-LABEL: func.func @basic_f16_matmul
func.func @basic_f16_matmul(%arg0: tensor<64x128xf16>) -> tensor<64x256xf16> {
  // Constant weight (required for macro mode)
  %weights = arith.constant dense<1.0> : tensor<128x256xf16>
  %init = tensor.empty() : tensor<64x256xf16>

  // CHECK-NOT: linalg.matmul
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x256xf16>
  // CHECK: %[[RESULT:.*]] = hexkl.matmul ins(%arg0, %{{.*}} : tensor<64x128xf16>, tensor<128x256xf16>) outs(%[[EMPTY]] : tensor<64x256xf16>)
  %result = linalg.matmul ins(%arg0, %weights : tensor<64x128xf16>, tensor<128x256xf16>)
                          outs(%init : tensor<64x256xf16>) -> tensor<64x256xf16>

  // CHECK: return %[[RESULT]]
  return %result : tensor<64x256xf16>
}

// -----

// ============================================================================
// Test 2: f16 inputs with f32 output (should convert with upcast)
// ============================================================================

// CHECK-LABEL: func.func @f16_to_f32_matmul
func.func @f16_to_f32_matmul(%arg0: tensor<32x64xf16>) -> tensor<32x128xf32> {
  %weights = arith.constant dense<2.0> : tensor<64x128xf16>
  %init = tensor.empty() : tensor<32x128xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: %[[F16_EMPTY:.*]] = tensor.empty() : tensor<32x128xf16>
  // CHECK: %[[HEXKL:.*]] = hexkl.matmul ins(%arg0, %{{.*}} : tensor<32x64xf16>, tensor<64x128xf16>) outs(%[[F16_EMPTY]] : tensor<32x128xf16>)

  // CHECK: %[[F32_EMPTY:.*]] = tensor.empty() : tensor<32x128xf32>
  // CHECK: %[[CAST:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[HEXKL]] : tensor<32x128xf16>)
  // CHECK-SAME: outs(%[[F32_EMPTY]] : tensor<32x128xf32>)
  // CHECK: arith.extf
  %result = linalg.matmul ins(%arg0, %weights : tensor<32x64xf16>, tensor<64x128xf16>)
                          outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>

  // CHECK: return %[[CAST]]
  return %result : tensor<32x128xf32>
}

// -----

// ============================================================================
// Test 3: Non-constant weights (should NOT convert in macro mode)
// ============================================================================

// CHECK-LABEL: func.func @non_constant_weights
func.func @non_constant_weights(%arg0: tensor<64x128xf16>, %weights: tensor<128x256xf16>) -> tensor<64x256xf16> {
  %init = tensor.empty() : tensor<64x256xf16>

  // Should NOT convert (weights are not constant)
  // CHECK: linalg.matmul
  // CHECK-NOT: hexkl.matmul
  %result = linalg.matmul ins(%arg0, %weights : tensor<64x128xf16>, tensor<128x256xf16>)
                          outs(%init : tensor<64x256xf16>) -> tensor<64x256xf16>

  return %result : tensor<64x256xf16>
}

// -----

// ============================================================================
// Test 4: Dimension not multiple of 32 (should NOT convert)
// ============================================================================

// CHECK-LABEL: func.func @non_aligned_dimensions
func.func @non_aligned_dimensions(%arg0: tensor<33x64xf16>) -> tensor<33x128xf16> {
  %weights = arith.constant dense<1.0> : tensor<64x128xf16>
  %init = tensor.empty() : tensor<33x128xf16>

  // Should NOT convert (M=33 is not multiple of 32)
  // CHECK: linalg.matmul
  // CHECK-NOT: hexkl.matmul
  %result = linalg.matmul ins(%arg0, %weights : tensor<33x64xf16>, tensor<64x128xf16>)
                          outs(%init : tensor<33x128xf16>) -> tensor<33x128xf16>

  return %result : tensor<33x128xf16>
}

// -----

// ============================================================================
// Test 5: Dimension at maximum limits (should convert)
// ============================================================================

// CHECK-LABEL: func.func @dimension_at_limits
func.func @dimension_at_limits(%arg0: tensor<1600x64xf16>) -> tensor<1600x5120xf16> {
  %weights = arith.constant dense<1.0> : tensor<64x5120xf16>
  %init = tensor.empty() : tensor<1600x5120xf16>

  // Should convert (M=1600 is at the limit, N=5120 is at the limit)
  // CHECK-NOT: linalg.matmul
  // CHECK: hexkl.matmul
  %result = linalg.matmul ins(%arg0, %weights : tensor<1600x64xf16>, tensor<64x5120xf16>)
                          outs(%init : tensor<1600x5120xf16>) -> tensor<1600x5120xf16>

  return %result : tensor<1600x5120xf16>
}

// -----

// ============================================================================
// Test 6: K dimension exceeds maximum (should NOT convert)
// ============================================================================

// CHECK-LABEL: func.func @k_exceeds_limit
func.func @k_exceeds_limit(%arg0: tensor<64x76064xf16>) -> tensor<64x128xf16> {
  %weights = arith.constant dense<1.0> : tensor<76064x128xf16>
  %init = tensor.empty() : tensor<64x128xf16>

  // Should NOT convert (K=76064 exceeds MAX_N_INNER=76030)
  // CHECK: linalg.matmul
  // CHECK-NOT: hexkl.matmul
  %result = linalg.matmul ins(%arg0, %weights : tensor<64x76064xf16>, tensor<76064x128xf16>)
                          outs(%init : tensor<64x128xf16>) -> tensor<64x128xf16>

  return %result : tensor<64x128xf16>
}
