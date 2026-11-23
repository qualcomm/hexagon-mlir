//===- specialize-tanh-gelu-f16.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -specialize-transcendentals -canonicalize | FileCheck %s

// Test that the pass matches tanh expanded with f16 precision-promotion casts.
//
// The front-end emits exp in f32 for precision but performs sub/add/mul in f16.
// The resulting IR has cast-only generics (extf / truncf) interleaved between
// the structural ops, which previously prevented pattern matching.
//
// Pattern (Form A, all-f16 with f32 exp):
//   u               : f16 tensor
//   exp(-2u)        : f32  (via extf → exp → truncf back to f16)
//   (1 - exp(-2u))  : f16  sub
//   (1 + exp(-2u))  : f16  add
//   div             : f32  (extf of sub and add, then divf)
//
// After the pass the divf is replaced by tanh(extf(u) : f32) and the dead
// intermediate chain is cleaned up by canonicalize.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @tanh_f16_cast_chain
// CHECK-NOT:   arith.divf
// CHECK:       arith.extf %{{.*}} : f16 to f32
// CHECK:       math.tanh
// CHECK:       arith.addf

func.func @tanh_f16_cast_chain(%arg0: tensor<1024xf16>) -> tensor<1024xf32> {
  %cst_neg2 = arith.constant -2.000000e+00 : f16
  %cst_one_f16 = arith.constant 1.000000e+00 : f16
  %cst_one_f32 = arith.constant 1.000000e+00 : f32

  %empty_f16 = tensor.empty() : tensor<1024xf16>
  %empty_f32 = tensor.empty() : tensor<1024xf32>

  %fill_neg2  = linalg.fill ins(%cst_neg2    : f16) outs(%empty_f16 : tensor<1024xf16>) -> tensor<1024xf16>
  %fill_one16 = linalg.fill ins(%cst_one_f16 : f16) outs(%empty_f16 : tensor<1024xf16>) -> tensor<1024xf16>
  %fill_one32 = linalg.fill ins(%cst_one_f32 : f32) outs(%empty_f32 : tensor<1024xf32>) -> tensor<1024xf32>

  // -2 * u  [f16]
  %neg2u = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%arg0, %fill_neg2 : tensor<1024xf16>, tensor<1024xf16>) outs(%empty_f16 : tensor<1024xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %v = arith.mulf %in, %in_1 : f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // extf(-2u)  [f16 → f32]
  %neg2u_f32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%neg2u : tensor<1024xf16>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // exp(-2u)  [f32]
  %exp_f32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%neg2u_f32 : tensor<1024xf32>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.exp %in : f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // truncf(exp(-2u))  [f32 → f16]
  %exp_f16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%exp_f32 : tensor<1024xf32>) outs(%empty_f16 : tensor<1024xf16>) {
  ^bb0(%in: f32, %out: f16):
    %v = arith.truncf %in : f32 to f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // 1 - exp(-2u)  [f16]
  %sub_f16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%fill_one16, %exp_f16 : tensor<1024xf16>, tensor<1024xf16>) outs(%empty_f16 : tensor<1024xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %v = arith.subf %in, %in_1 : f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // exp(-2u) + 1  [f16]
  %add_f16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%exp_f16, %fill_one16 : tensor<1024xf16>, tensor<1024xf16>) outs(%empty_f16 : tensor<1024xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %v = arith.addf %in, %in_1 : f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // extf(1 - exp(-2u))  [f16 → f32]
  %sub_f32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%sub_f16 : tensor<1024xf16>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // extf(exp(-2u) + 1)  [f16 → f32]
  %add_f32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%add_f16 : tensor<1024xf16>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // (1 - exp(-2u)) / (1 + exp(-2u))  →  replaced by tanh(u) [f32]
  %div_f32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%sub_f32, %add_f32 : tensor<1024xf32>, tensor<1024xf32>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %v = arith.divf %in, %in_1 : f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // tanh(u) + 1  — verifies downstream use of the replaced op is preserved
  %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%div_f32, %fill_one32 : tensor<1024xf32>, tensor<1024xf32>) outs(%empty_f32 : tensor<1024xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %v = arith.addf %in, %in_1 : f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  return %result : tensor<1024xf32>
}
