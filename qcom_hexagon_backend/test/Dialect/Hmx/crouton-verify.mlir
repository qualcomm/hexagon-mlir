//===- crouton-verify.mlir - #hmx.crouton encoding verifier ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The `#hmx.crouton` tensor encoding is a `VerifiableTensorEncoding`, so
// `RankedTensorType::verify` rejects an illegal crouton type as soon as it is
// written; no per-op verifier repeats the check. These are the five conditions
// from docs/hmx/layout-representation-survey.md section 4.2 plus the
// parameter-only checks.
//
// The cases are function *declarations* (`func.func private @...`) on purpose:
// an encoding error is a parse-time diagnostic, and only a bodyless function
// declaration recovers cleanly so that `-split-input-file` can continue to the
// next case.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics
//===----------------------------------------------------------------------===//

//--- the crouton array is a 2D grid of croutons: rank 5
// expected-error @+1 {{rank-5}}
func.func private @rank4(%a: tensor<2x4x16x32xf16, #hmx.crouton<logical = [64, 128]>>)

// -----

//--- the engine only reads f16 (a wider result is widened outside the layout)
// expected-error @+1 {{f16}}
func.func private @f32_elements(%a: tensor<2x4x16x32x2xf32, #hmx.crouton<logical = [64, 128]>>)

// -----

//--- the trailing dims are exactly one crouton: 16, 32, 2
// expected-error @+1 {{must end in a crouton}}
func.func private @bad_tail(%a: tensor<2x4x16x32x4xf16, #hmx.crouton<logical = [64, 128]>>)

// -----

//--- an empty grid is not a crouton array
// expected-error @+1 {{non-empty}}
func.func private @empty_grid(%a: tensor<0x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>)

// -----

//--- the logical shape must agree with the physical grid
// expected-error @+1 {{must equal grid}}
func.func private @logical_mismatch(%a: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 96]>>)

// -----

//--- logical is a 2D extent pair
// expected-error @+1 {{exactly two logical dimensions}}
func.func private @logical_arity(%a: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128, 128]>>)

// -----

//--- logical extents are positive
// expected-error @+1 {{positive}}
func.func private @logical_negative(%a: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [-64, 128]>>)

// -----

//--- and 32-aligned: the engine's hard prerequisite, caught at the type
// expected-error @+1 {{multiples of the crouton tile}}
func.func private @logical_unaligned(%a: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [60, 128]>>)

// -----

//=== the memref-side carrier: #hmx.crouton_memref_layout ====
//
// `MemRefType::verify` calls the layout's `verifyLayout` whenever the type is
// formed, so the same conditions are enforced on the memref side. (The f16
// element check has no memref-side counterpart: a layout never sees the
// element type.)

//--- the carrier is only legal on the rank-5 crouton array
// expected-error @+1 {{rank-5}}
func.func private @memref_layout_rank4(%a: memref<2x4x16x32xf16, #hmx.crouton_memref_layout<logical = [64, 128]>>)

// -----

//--- the tail must be a crouton, here too
// expected-error @+1 {{must end in a crouton}}
func.func private @memref_layout_bad_tail(%a: memref<2x4x16x32x4xf16, #hmx.crouton_memref_layout<logical = [64, 128]>>)

// -----

//--- the logical shape must agree with the physical grid, here too
// expected-error @+1 {{must equal grid}}
func.func private @memref_layout_logical_mismatch(%a: memref<2x4x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 96]>>)

// -----

//--- and the logical parameters carry the same contract as the encoding
// expected-error @+1 {{multiples of the crouton tile}}
func.func private @memref_layout_unaligned(%a: memref<2x4x16x32x2xf16, #hmx.crouton_memref_layout<logical = [60, 128]>>)
