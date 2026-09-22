//===- hvx-to-llvm.mlir - Lower hvx ops to LLVM ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -hvx-to-llvm | FileCheck %s
//===----------------------------------------------------------------------===//

// The intrinsic is declared once, with its exact signature (our LLVM selects
// llvm.hexagon.V6.vror.128B into V6_vror under UseHVX128B).
// CHECK: llvm.func @llvm.hexagon.V6.vror.128B(vector<32xi32>, i32) -> vector<32xi32>

// CHECK-LABEL: llvm.func @vror_f32
llvm.func @vror_f32(%v: vector<32xf32>) -> vector<32xf32> {
  // The lane-typed vector is bitcast to the intrinsic's <32 x i32> register
  // image, called, and bitcast back.
  // CHECK: %[[image:.*]] = llvm.bitcast %{{.*}} : vector<32xf32> to vector<32xi32>
  // CHECK: %[[amt:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[res:.*]] = llvm.call @llvm.hexagon.V6.vror.128B(%[[image]], %[[amt]]) : (vector<32xi32>, i32) -> vector<32xi32>
  // CHECK: llvm.bitcast %[[res]] : vector<32xi32> to vector<32xf32>
  %0 = hvx.vror %v, 64 : vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// -----

// A half-register f16 rotate goes through the same register image.
// CHECK-LABEL: llvm.func @vror_f16
llvm.func @vror_f16(%v: vector<64xf16>) -> vector<64xf16> {
  // CHECK: llvm.call @llvm.hexagon.V6.vror.128B
  %0 = hvx.vror %v, 32 : vector<64xf16>
  llvm.return %0 : vector<64xf16>
}
