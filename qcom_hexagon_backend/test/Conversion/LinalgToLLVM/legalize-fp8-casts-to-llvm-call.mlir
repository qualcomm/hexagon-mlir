//===- legalize-fp8-casts-to-llvm-call.mlir -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -legalize-fp8-casts-to-llvm-call -canonicalize | FileCheck %s

// This tests replacement of builtin.unrealized_conversion_cast ..vector<128xi8> to vector<128xf8E4M3FN> with e.g. @convert_fp8_fp16_fnuz

module {
  // CHECK-LABEL: llvm.func @kernel_extf_vec
  // CHECK: llvm.bitcast
  // CHECK: llvm.call @convert_fp8_fp16_fnuz
  // CHECK-NOT: arith.extf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_extf_vec(%arg0: vector<128xi8>) -> vector<128xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<128xf8E4M3FNUZ>
    %1 = arith.extf %0 : vector<128xf8E4M3FNUZ> to vector<128xf16>
    llvm.return %1 : vector<128xf16>
  }

  // CHECK-LABEL: llvm.func @kernel_extf_vec_fn
  // CHECK: llvm.bitcast
  // CHECK: llvm.call @convert_fp8_fp16_fn
  // CHECK-NOT: arith.extf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_extf_vec_fn(%arg0: vector<128xi8>) -> vector<128xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<128xf8E4M3FN>
    %1 = arith.extf %0 : vector<128xf8E4M3FN> to vector<128xf16>
    llvm.return %1 : vector<128xf16>
  }

  // CHECK-LABEL: llvm.func @kernel_trunc_vec
  // CHECK: llvm.bitcast
  // CHECK: llvm.call @convert_fp16_fp8_fnuz
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_trunc_vec(%arg0: vector<128xf16>) -> vector<128xi8> {
    %0 = arith.truncf %arg0 : vector<128xf16> to vector<128xf8E4M3FNUZ>
    %1 = builtin.unrealized_conversion_cast %0 : vector<128xf8E4M3FNUZ> to vector<128xi8>
    llvm.return %1 : vector<128xi8>
  }

  // CHECK-LABEL: llvm.func @kernel_trunc_vec_fn
  // CHECK: llvm.bitcast
  // CHECK: llvm.call @convert_fp16_fp8_fn
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_trunc_vec_fn(%arg0: vector<128xf16>) -> vector<128xi8> {
    %0 = arith.truncf %arg0 : vector<128xf16> to vector<128xf8E4M3FN>
    %1 = builtin.unrealized_conversion_cast %0 : vector<128xf8E4M3FN> to vector<128xi8>
    llvm.return %1 : vector<128xi8>
  }
}
