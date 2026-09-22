//===- verify.mlir - Hvx dialect verifier negative cases -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

//--- !!! One HVX vector register is 128 bytes; anything else is not vror-shaped
func.func @half_register(%v: vector<32xf16>) -> vector<32xf16> {
  // expected-error @+1 {{operand must be one 128-byte vector, got 64 bytes}}
  %0 = hvx.vror %v, 32 : vector<32xf16>
  return %0 : vector<32xf16>
}

//--- !!! The instruction rotates by Rt mod 128; a full-width amount is a
//--- no-op, so the verifier rejects it rather than accepting dead work
func.func @full_width_rotate(%v: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{rotate amount must be in [0, 128) bytes, got 128}}
  %0 = hvx.vror %v, 128 : vector<32xf32>
  return %0 : vector<32xf32>
}
