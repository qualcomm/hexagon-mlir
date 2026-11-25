//===- pow-expand-positive-base.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -expand-math-ops | FileCheck %s

// Verify that math.powf is expanded to exp(exponent * log(base)) by
// expand-math-ops when the base is provably positive, and left intact when
// the base sign is unknown (falling through to UseHexagonRoutines later).

// CHECK-LABEL: func.func @pow_with_exp_base
// CHECK-NOT: math.powf
// CHECK: [[BASE:%.+]] = math.exp %arg0
// CHECK: [[LOG:%.+]] = math.log [[BASE]]
// CHECK: [[SCALED:%.+]] = arith.mulf %arg1, [[LOG]]
// CHECK: math.exp [[SCALED]]
func.func @pow_with_exp_base(%x: vector<32xf32>, %y: vector<32xf32>) -> vector<32xf32> {
  %base = math.exp %x : vector<32xf32>
  %result = math.powf %base, %y : vector<32xf32>
  return %result : vector<32xf32>
}

// CHECK-LABEL: func.func @pow_with_abs_base
// CHECK-NOT: math.powf
// CHECK: [[ABS:%.+]] = math.absf %arg0
// CHECK: [[LOG:%.+]] = math.log [[ABS]]
// CHECK: [[SCALED:%.+]] = arith.mulf %arg1, [[LOG]]
// CHECK: math.exp [[SCALED]]
func.func @pow_with_abs_base(%x: vector<32xf32>, %y: vector<32xf32>) -> vector<32xf32> {
  %base = math.absf %x : vector<32xf32>
  %result = math.powf %base, %y : vector<32xf32>
  return %result : vector<32xf32>
}

// Unknown base (function argument): must NOT be expanded to avoid NaN for
// negative bases. The op is left intact for UseHexagonRoutines to handle.
// CHECK-LABEL: func.func @pow_unknown_base
// CHECK: math.powf %arg0, %arg1
// CHECK-NOT: math.log
func.func @pow_unknown_base(%base: vector<32xf32>, %y: vector<32xf32>) -> vector<32xf32> {
  %result = math.powf %base, %y : vector<32xf32>
  return %result : vector<32xf32>
}
