//===- replace_linalgop_library_call.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -linalg-to-llvm  | FileCheck %s

// CHECK-LABEL: llvm.func private @foo
// CHECK-LABEL: @matmul
// CHECK: llvm.call @foo
func.func @matmul(%arg0: tensor<64x256xf16>,
                  %arg1: tensor<256x128xf16>,
                  %arg2: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %0 = linalg.matmul {library_call = "foo"} ins (%arg0, %arg1: tensor<64x256xf16>, tensor<256x128xf16>)
                outs (%arg2: tensor<64x128xf16>) -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}
