//===- weight-resident-runtime-pipeline.mlir - runtime weight end to end ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The runtime-weight anchor of matmul-to-hmx (W arrives as a function argument),
// run through the whole LinalgToLLVM pipeline with the P2 gate on. The kernel no
// longer emits the per-launch `hmx_pack_weight` leaf: the weight is a resident
// VTCM buffer whose one copy happens inside the runtime on the first launch.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-weight-resident=true})' | FileCheck %s
//===----------------------------------------------------------------------===//

// The runtime gets the argument's address; no per-launch pack leaf is emitted.
// CHECK: llvm.func @hexagon_runtime_weight_resident_dsp(i64, i32) -> !llvm.ptr
// CHECK-LABEL: llvm.func @runtime_weight
// CHECK: llvm.call @hexagon_runtime_weight_resident_dsp
// CHECK-NOT: llvm.call @hmx_pack_weight_f16
module {
  func.func @runtime_weight(%a: tensor<64x64xf16>, %w: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
