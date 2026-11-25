//===- specialize-tanh-gelu-f32.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -specialize-transcendentals -canonicalize | FileCheck %s

// Test derived from the gelu_kernel IR.
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))
//
// The tanh itself is encoded as:
//   (1 - exp(-2u)) / (1 + exp(-2u))
//
// After the pass the div is replaced by linalg.tanh(u), and the dead
// intermediate chain (mul_neg2u, exp, sub, add) is cleaned up by canonicalize.
// The rest of the gelu computation (scaling by 0.5*x, adding 1) is preserved.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @gelu_kernel
// CHECK-NOT:   arith.divf
// CHECK:       math.tanh
// CHECK:       arith.addf
// CHECK:       arith.mulf

func.func @gelu_kernel(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %cst    = arith.constant -2.000000e+00 : f32
  %cst_0  = arith.constant  1.000000e+00 : f32
  %cst_1  = arith.constant  3.567700e-02 : f32
  %cst_2  = arith.constant  7.978850e-01 : f32
  %cst_3  = arith.constant  5.000000e-01 : f32
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.fill ins(%cst   : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>  // fill(-2)
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>  // fill(1)
  %3 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>  // fill(0.035677)
  %4 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>  // fill(0.797885)
  %5 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>  // fill(0.5)

  // 0.5 * x
  %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%arg0, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 0.797885 * x
  %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%arg0, %4 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 0.035677 * x
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%arg0, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 0.035677 * x^2
  %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%11, %arg0 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 0.035677 * x^3
  %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%12, %arg0 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // u = 0.797885*x + 0.035677*x^3  (tanh argument)
  %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%10, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.addf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // -2 * u
  %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%14, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // exp(-2u)
  %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%15 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %22 = math.exp %in : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 1 - exp(-2u)   ← fill(1) is operand 0: Form A requires multiplier = -2 ✓
  %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%2, %16 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.subf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // exp(-2u) + 1   ← add shares the same fill(1) and exp SSA values ✓
  %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%16, %2 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.addf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // (1 - exp(-2u)) / (1 + exp(-2u))  →  replaced by linalg.tanh(u)
  %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%17, %18 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.divf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // tanh(u) + 1
  %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%19, %2 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.addf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  // 0.5 * x * (tanh(u) + 1)
  %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%9, %20 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %22 = arith.mulf %in, %in_5 : f32
    linalg.yield %22 : f32
  } -> tensor<1024xf32>

  return %21 : tensor<1024xf32>
}
