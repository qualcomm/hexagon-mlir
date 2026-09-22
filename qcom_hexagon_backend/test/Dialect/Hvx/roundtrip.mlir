//===- roundtrip.mlir - Hvx dialect roundtrip ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @vror_f32
func.func @vror_f32(%v: vector<32xf32>) -> vector<32xf32> {
  // CHECK: hvx.vror %{{.*}}, 64 : vector<32xf32>
  %0 = hvx.vror %v, 64 : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// The op is element-type agnostic: the instruction permutes bytes, so any
// lane type that fills exactly one 128-byte register rides on it.
// CHECK-LABEL: func.func @vror_f16_i8
func.func @vror_f16_i8(%a: vector<64xf16>, %b: vector<128xi8>) -> (vector<64xf16>, vector<128xi8>) {
  // CHECK: hvx.vror %{{.*}}, 2 : vector<64xf16>
  %0 = hvx.vror %a, 2 : vector<64xf16>
  // CHECK: hvx.vror %{{.*}}, 127 : vector<128xi8>
  %1 = hvx.vror %b, 127 : vector<128xi8>
  return %0, %1 : vector<64xf16>, vector<128xi8>
}
