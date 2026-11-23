//===- pow_hexagon_routines.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(expand-math-ops,convert-math-to-llvm),convert-func-to-llvm,llvm-intr-to-hexagon-routines)' | FileCheck %s

// Verify that math.powf is NOT expanded to exp(y*log(x)) by expand-math-ops,
// and instead flows through to UseHexagonRoutines as a two-operand call to
// _hexagon_runtime_pow__vf / _hexagon_runtime_pow__vhf.

// CHECK-LABEL: llvm.func @test_pow_f32
// CHECK-NOT: math.exp
// CHECK-NOT: math.log
// CHECK: [[B:%.+]] = llvm.bitcast %arg0 : vector<32xf32> to vector<32xi32>
// CHECK: [[E:%.+]] = llvm.bitcast %arg1 : vector<32xf32> to vector<32xi32>
// CHECK: [[R:%.+]] = llvm.call @_hexagon_runtime_pow__vf([[B]], [[E]])
// CHECK: llvm.bitcast [[R]] : vector<32xi32> to vector<32xf32>
func.func @test_pow_f32(%base: vector<32xf32>, %exp: vector<32xf32>) -> vector<32xf32> {
  %result = math.powf %base, %exp : vector<32xf32>
  return %result : vector<32xf32>
}

// CHECK-LABEL: llvm.func @test_pow_f16
// CHECK: [[B16:%.+]] = llvm.bitcast %arg0 : vector<64xf16> to vector<32xi32>
// CHECK: [[E16:%.+]] = llvm.bitcast %arg1 : vector<64xf16> to vector<32xi32>
// CHECK: [[R16:%.+]] = llvm.call @_hexagon_runtime_pow__vhf([[B16]], [[E16]])
// CHECK: llvm.bitcast [[R16]] : vector<32xi32> to vector<64xf16>
func.func @test_pow_f16(%base: vector<64xf16>, %exp: vector<64xf16>) -> vector<64xf16> {
  %result = math.powf %base, %exp : vector<64xf16>
  return %result : vector<64xf16>
}
