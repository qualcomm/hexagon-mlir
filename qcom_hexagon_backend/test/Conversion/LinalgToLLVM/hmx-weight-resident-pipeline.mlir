//===- weight-resident-pipeline.mlir - constant weight end to end ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The constant-weight anchor of matmul-to-hmx (a compile-time weight), run
// through the whole LinalgToLLVM pipeline. Before residency the prepacked
// constant stayed a DDR `memref.global` and the tile level rejected it; now the
// weight reaches the engine as a resident VTCM buffer whose one copy happens
// inside the runtime on the first launch.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The runtime gets the address of the prepacked constant, which by now is an
// LLVM global. No `hexagon_runtime_alloc_1d_dsp` for the weight: it is not
// allocated per launch.
// CHECK: llvm.func @hexagon_runtime_weight_resident_dsp(i64, i32) -> !llvm.ptr
// CHECK: llvm.mlir.global {{.*}} @__constant_2x2x16x32x2xf16
// CHECK-LABEL: llvm.func @constant_weight
// CHECK: llvm.mlir.addressof @__constant_2x2x16x32x2xf16
// CHECK: llvm.call @hexagon_runtime_weight_resident_dsp
// The weight is filled once, by the runtime; the kernel emits no copy for it.
// CHECK-NOT: hexagon_runtime_copy_dsp{{.*}}__constant
module {
  func.func @constant_weight(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %w = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
