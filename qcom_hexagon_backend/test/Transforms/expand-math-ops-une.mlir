//===- expand-math-ops-une.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -expand-math-ops | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(expand-math-ops{expand-une-on-vectors=false}))' | FileCheck %s --check-prefix=DISABLED

// -----

// une(x, x) on f32 vector: isNaN fast path.
// Both operands are the same SSA value so one(x,x) is always false;
// result reduces to (bitcast(x) & 0x7FFFFFFF) > 0x7F800000.

// CHECK-LABEL: func.func @isnan_vec_f32
// CHECK-NOT:   arith.cmpf
// CHECK:       arith.constant dense<2139095040> : vector<32xi32>
// CHECK:       arith.constant dense<2147483647> : vector<32xi32>
// CHECK:       arith.bitcast {{.*}} : vector<32xf32> to vector<32xi32>
// CHECK:       arith.andi
// CHECK:       arith.cmpi ugt
// DISABLED-LABEL: func.func @isnan_vec_f32
// DISABLED:    arith.cmpf une
func.func @isnan_vec_f32(%x: vector<32xf32>) -> vector<32xi1> {
  %r = arith.cmpf une, %x, %x : vector<32xf32>
  return %r : vector<32xi1>
}

// -----

// une(x, x) on f16 vector: same fast path with f16 bit constants
// (mask = 0x7FFF = 32767, inf = 0x7C00 = 31744).

// CHECK-LABEL: func.func @isnan_vec_f16
// CHECK-NOT:   arith.cmpf
// CHECK:       arith.constant dense<31744> : vector<32xi16>
// CHECK:       arith.constant dense<32767> : vector<32xi16>
// CHECK:       arith.bitcast {{.*}} : vector<32xf16> to vector<32xi16>
// CHECK:       arith.andi
// CHECK:       arith.cmpi ugt
// DISABLED-LABEL: func.func @isnan_vec_f16
// DISABLED:    arith.cmpf une
func.func @isnan_vec_f16(%x: vector<32xf16>) -> vector<32xi1> {
  %r = arith.cmpf une, %x, %x : vector<32xf16>
  return %r : vector<32xi1>
}

// -----

// une(a, b) on f32 vector: general case.
// une(a,b) = isNaN(a) | isNaN(b) | one(a,b); one is ordered so HVX handles it.

// CHECK-LABEL: func.func @une_vec_f32
// CHECK-NOT:   arith.cmpf une
// CHECK:       arith.bitcast {{.*}} : vector<32xf32> to vector<32xi32>
// CHECK:       arith.bitcast {{.*}} : vector<32xf32> to vector<32xi32>
// CHECK:       arith.cmpf one, {{.*}} : vector<32xf32>
// CHECK:       arith.ori
// CHECK:       arith.ori
// DISABLED-LABEL: func.func @une_vec_f32
// DISABLED:    arith.cmpf une
func.func @une_vec_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xi1> {
  %r = arith.cmpf une, %a, %b : vector<32xf32>
  return %r : vector<32xi1>
}

// -----

// Scalar une is not touched — Hexagon scalar units handle unordered compares.

// CHECK-LABEL: func.func @scalar_une_unchanged
// CHECK:       arith.cmpf une
// DISABLED-LABEL: func.func @scalar_une_unchanged
// DISABLED:    arith.cmpf une
func.func @scalar_une_unchanged(%x: f32) -> i1 {
  %r = arith.cmpf une, %x, %x : f32
  return %r : i1
}
