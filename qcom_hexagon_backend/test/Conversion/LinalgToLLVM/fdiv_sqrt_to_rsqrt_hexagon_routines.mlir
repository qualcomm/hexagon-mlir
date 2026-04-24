// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' | FileCheck %s -check-prefixes=CHECK

llvm.func @test_fdiv_sqrt_pattern(%arg0: vector<64xf16>, %arg1: vector<64xf32>) -> vector<64xf16> {

  // Test f16 vector: 1.0 / sqrt(x) should become rsqrt(x) runtime call
  // CHECK-LABEL: @test_fdiv_sqrt_pattern(
  // CHECK-SAME:  %arg0: vector<64xf16>, %arg1: vector<64xf32>) -> vector<64xf16>
  // CHECK: [[BITCAST:%.+]] = llvm.bitcast %arg0 : vector<64xf16> to vector<32xi32>
  // CHECK-NEXT: [[INTRINCALL:%.+]] = llvm.call @_hexagon_runtime_rsqrt__vhf([[BITCAST]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK-NEXT: [[RSQRT:%.+]] = llvm.bitcast [[INTRINCALL]] : vector<32xi32> to vector<64xf16>

  %sqrt_f16 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf16>) -> vector<64xf16>
  %one_f16 = llvm.mlir.constant(dense<1.0> : vector<64xf16>) : vector<64xf16>
  %rsqrt_f16 = llvm.fdiv %one_f16, %sqrt_f16 {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>

  // Test f32 vector: 1.0 / sqrt(x) should become rsqrt(x) with vector splitting
  // CHECK: [[EXTRACTV:%.+]] = llvm.intr.vector.extract %arg1[0] : vector<32xf32> from vector<64xf32>
  // CHECK-NEXT: [[BITCAST1:%.+]] = llvm.bitcast [[EXTRACTV]] : vector<32xf32> to vector<32xi32>
  // CHECK-NEXT: [[RSQRT1:%.+]] = llvm.call @_hexagon_runtime_rsqrt__vf([[BITCAST1]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK-NEXT: [[BITCAST2:%.+]] = llvm.bitcast [[RSQRT1]] : vector<32xi32> to vector<32xf32>
  // CHECK: [[EXTRACTV2:%.+]] = llvm.intr.vector.extract %arg1[32] : vector<32xf32> from vector<64xf32>
  // CHECK-NEXT: [[BITCAST3:%.+]] = llvm.bitcast [[EXTRACTV2]] : vector<32xf32> to vector<32xi32>
  // CHECK-NEXT: [[RSQRT2:%.+]] = llvm.call @_hexagon_runtime_rsqrt__vf([[BITCAST3]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK-NEXT: [[BITCAST4:%.+]] = llvm.bitcast [[RSQRT2]] : vector<32xi32> to vector<32xf32>

  %sqrt_f32 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  %one_f32 = llvm.mlir.constant(dense<1.0> : vector<64xf32>) : vector<64xf32>
  %rsqrt_f32 = llvm.fdiv %one_f32, %sqrt_f32 {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf32>

  // CHECK: [[TRUNC:%.+]] = llvm.fptrunc {{%.+}} : vector<64xf32> to vector<64xf16>
  %trunc = llvm.fptrunc %rsqrt_f32 : vector<64xf32> to vector<64xf16>

  // CHECK: [[SUB:%.+]] = llvm.fsub [[RSQRT]], [[TRUNC]] {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
  %result = llvm.fsub %rsqrt_f16, %trunc {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>

  llvm.return %result : vector<64xf16>
}

llvm.func @test_fdiv_sqrt_pattern_scalar(%arg0: f16, %arg1: f32) -> f16 {

  // Test f16 scalar: 1.0 / sqrt(x) should become rsqrt(x) runtime call
  // CHECK-LABEL: @test_fdiv_sqrt_pattern_scalar(
  // CHECK-SAME:  %arg0: f16, %arg1: f32) -> f16
  // CHECK: %[[FPEXT:.+]] = llvm.fpext %arg0 : f16 to f32
  // CHECK-NEXT: %[[RSQRT_F16:.+]] = llvm.call @qhmath_rsqrt_f(%[[FPEXT]]) : (f32) -> f32
  // CHECK-NEXT: %[[TRUNC_F16:.+]] = llvm.fptrunc %[[RSQRT_F16]] : f32 to f16

  %sqrt_f16 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f16) -> f16
  %one_f16 = llvm.mlir.constant(1.0 : f16) : f16
  %rsqrt_f16 = llvm.fdiv %one_f16, %sqrt_f16 {fastmathFlags = #llvm.fastmath<fast>} : f16

  // Test f32 scalar: 1.0 / sqrt(x) should become rsqrt(x) runtime call
  // CHECK: %[[RSQRT_F32:.+]] = llvm.call @qhmath_rsqrt_f(%arg1) : (f32) -> f32

  %sqrt_f32 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %one_f32 = llvm.mlir.constant(1.0 : f32) : f32
  %rsqrt_f32 = llvm.fdiv %one_f32, %sqrt_f32 {fastmathFlags = #llvm.fastmath<fast>} : f32

  // CHECK: %[[TRUNC_F32:.+]] = llvm.fptrunc %[[RSQRT_F32]] : f32 to f16
  %trunc = llvm.fptrunc %rsqrt_f32 : f32 to f16

  // CHECK: [[SUB:%.+]] = llvm.fsub %[[TRUNC_F16]], %[[TRUNC_F32]] {fastmathFlags = #llvm.fastmath<fast>} : f16
  %result = llvm.fsub %rsqrt_f16, %trunc {fastmathFlags = #llvm.fastmath<fast>} : f16

  llvm.return %result : f16
}
