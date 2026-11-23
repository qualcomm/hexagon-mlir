//===- fold-mulf-by-zero-ieee.mlir ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(fold-mulf-by-zero))' | FileCheck %s

// Verify that FoldMulFByZero produces IEEE-754-correct results for
// inf * 0.0 (must be NaN) in addition to NaN propagation and signed zero.

// Scalar case.
// CHECK-LABEL: func.func @mulf_zero_scalar
// CHECK-SAME:    (%[[X:.*]]: f32) -> f32
// CHECK-DAG:     %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-DAG:     %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[IS_NAN:.*]] = arith.cmpf uno, %[[X]], %[[X]] : f32
// CHECK:         %[[ABS:.*]] = math.absf %[[X]] : f32
// CHECK:         %[[IS_INF:.*]] = arith.cmpf oeq, %[[ABS]], %[[INF]] : f32
// CHECK:         %[[SZERO:.*]] = math.copysign %[[ZERO]], %[[X]] : f32
// CHECK:         %[[SEL_INF:.*]] = arith.select %[[IS_INF]], %[[NAN]], %[[SZERO]] : f32
// CHECK:         %[[RES:.*]] = arith.select %[[IS_NAN]], %[[X]], %[[SEL_INF]] : f32
// CHECK:         return %[[RES]] : f32
func.func @mulf_zero_scalar(%arg0 : f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = arith.mulf %arg0, %zero : f32
  return %r : f32
}

// Vector case.
// CHECK-LABEL: func.func @mulf_zero_vector
// CHECK-SAME:    (%[[X:.*]]: vector<32xf32>) -> vector<32xf32>
// CHECK-DAG:     %[[NAN:.*]] = arith.constant dense<0x7FC00000> : vector<32xf32>
// CHECK-DAG:     %[[INF:.*]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK:         %[[IS_NAN:.*]] = arith.cmpf uno, %[[X]], %[[X]] : vector<32xf32>
// CHECK:         %[[ABS:.*]] = math.absf %[[X]] : vector<32xf32>
// CHECK:         %[[IS_INF:.*]] = arith.cmpf oeq, %[[ABS]], %[[INF]] : vector<32xf32>
// CHECK:         %[[SZERO:.*]] = math.copysign %[[ZERO]], %[[X]] : vector<32xf32>
// CHECK:         %[[SEL_INF:.*]] = arith.select %[[IS_INF]], %[[NAN]], %[[SZERO]] : vector<32xi1>, vector<32xf32>
// CHECK:         %[[RES:.*]] = arith.select %[[IS_NAN]], %[[X]], %[[SEL_INF]] : vector<32xi1>, vector<32xf32>
// CHECK:         return %[[RES]] : vector<32xf32>
func.func @mulf_zero_vector(%arg0 : vector<32xf32>) -> vector<32xf32> {
  %zero = arith.constant dense<0.0> : vector<32xf32>
  %r = arith.mulf %arg0, %zero : vector<32xf32>
  return %r : vector<32xf32>
}
