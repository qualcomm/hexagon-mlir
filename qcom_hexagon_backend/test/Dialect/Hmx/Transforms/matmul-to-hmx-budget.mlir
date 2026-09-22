//===- matmul-to-hmx-budget.mlir - the VTCM budget gate --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The second dot processed sees the first dot's residency: with room for one
// bridge under the budget, exactly one matmul is attributed and the other stays
// row-major. A 64x64x64 bridge is 24576 bytes, so the budget below fits one but
// not two -- checked against committed bytes, not an empty budget. The CHECKs
// are order-free on purpose: whichever dot the greedy driver attributes second
// is the one that is refused.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=30000}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @second_dot_sees_residency
// CHECK-DAG: hmx.matmul
// CHECK-DAG: hmx.unpack_acc
// CHECK-DAG: linalg.matmul
// CHECK-NOT: hmx.matmul
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: linalg.matmul
func.func @second_dot_sees_residency(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                                     %c: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %z1 = arith.constant 0.000000e+00 : f16
  %i1 = linalg.fill ins(%z1 : f16) outs(%e1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%m0, %c : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}
