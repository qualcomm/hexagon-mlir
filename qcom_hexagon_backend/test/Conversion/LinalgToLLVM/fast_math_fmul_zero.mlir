// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(fold-mulf-by-zero))' | FileCheck %s

// CHECK-LABEL: func.func @test_mulf_fold
func.func @test_mulf_fold(%arg0 : f32) -> (f32, f32) {
  // CHECK:  %[[C0:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:  %[[C0n:.+]] = arith.constant -0.000000e+00 : f32
  // CHECK-NEXT: return %[[C0]], %[[C0n]]
  %c0 = arith.constant 0.0 : f32
  %c0n = arith.constant -0.0 : f32
  %0 = arith.mulf %arg0, %c0 fastmath<nnan,nsz> : f32
  %1 = arith.mulf %arg0, %c0n fastmath<nnan,nsz> : f32
  return %0, %1 : f32, f32
}
