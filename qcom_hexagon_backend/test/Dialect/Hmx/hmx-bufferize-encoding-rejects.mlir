//===- hmx-bufferize-encoding-rejects.mlir - encoded alloc_tensor vs one-shot
//===- bufferize --------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Documents why `matmul-to-hmx` drops the `#hmx.crouton` encodings by default:
// an encoding that reaches one-shot bufferization through the *stock*
// `bufferization.alloc_tensor` is buffered into a fully-dynamic-strided
// `memref.alloc` without the symbol operands that layout requires, and the op
// verifier rejects the result. (The stock op's `getBufferType` ignores the
// encoding entirely; only the dialect's own `hmx.alloc_crouton` maps the
// encoding to `#hmx.crouton_memref_layout` -- see hmx-bufferize.mlir
// @crouton_alloc_keeps_layout and @memref_layout_carried.)
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries})' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @default_bufferize_rejects_encoded_alloc(
    %a: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
    %b: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>) -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>> {
  // expected-error @below {{'memref.alloc' op symbol operand count does not equal memref symbol count}}
  %crout = bufferization.alloc_tensor() {memory_space = 1 : i64}
      : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  %0 = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
                                  tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
                  outs(%crout : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
      -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  return %0 : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
}
