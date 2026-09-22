//===- matmul-to-hmx-remark.mlir - why an f16 matmul was skipped ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An f16 matmul that misses the engine's contract is a near miss, and the pass
// explains it. Every case here has operands the engine could take and an empty
// initialiser, so each one fails exactly one condition.
//
// Besides the per-op remark the pass aggregates one module-level warning per
// run: the production pipeline never shows remarks, so that warning is what a
// silently refused HMX path looks like there. It quotes the counts and the
// first refusal. Each module below is spelled explicitly so the warning has a
// line of its own to attach to (an implicit module gets a line-0 location).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=10000}))' -verify-diagnostics -split-input-file
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-allocator=false}))' -split-input-file 2>&1 | FileCheck %s --check-prefix=ENV
//
// With the VTCM allocator off the environment refusal fires before any shape
// question and leads the warning with the switch that closed the whole path:
// ENV: HMX disabled: this pipeline has no VTCM allocator (enableConvertToHexagonmem is off)
//===----------------------------------------------------------------------===//

// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4; matmul M=64, N=64, K=100, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @k_not_aligned(%a: tensor<64x100xf16>, %b: tensor<100x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  // expected-remark @+1 {{HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x100xf16>, tensor<100x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}

// -----

// expected-warning @+1 {{first refusal: HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4; matmul M=64, N=48, K=128}}
module {
func.func @n_not_aligned(%a: tensor<64x128xf16>, %b: tensor<128x48xf16>) -> tensor<64x48xf16> {
  %c = tensor.empty() : tensor<64x48xf16>
  // expected-remark @+1 {{HMX not applied}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x48xf16>) outs(%c : tensor<64x48xf16>) -> tensor<64x48xf16>
  return %0 : tensor<64x48xf16>
}
}

// -----

// Too few rows to form a tile.
// expected-warning @+1 {{first refusal: HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4; matmul M=2, N=64, K=64}}
module {
func.func @too_few_rows(%a: tensor<2x64xf16>, %b: tensor<64x64xf16>) -> tensor<2x64xf16> {
  %c = tensor.empty() : tensor<2x64xf16>
  // expected-remark @+1 {{M > 4}}
  %0 = linalg.matmul ins(%a, %b : tensor<2x64xf16>, tensor<64x64xf16>) outs(%c : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %0 : tensor<2x64xf16>
}
}

// -----

// A bridge that does not fit the remaining VTCM budget is left alone, loudly:
// a 64x64x64 bridge is 24576 bytes and its smallest M block (one 32-row tile,
// 16384 bytes) is still over the budget below, so no block fits and the op is
// refused rather than blocked.
// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: matmul (M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=10000 bytes); matmul M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @over_budget(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  // expected-remark @+1 {{HMX not applied: matmul (M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=10000 bytes)}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}
