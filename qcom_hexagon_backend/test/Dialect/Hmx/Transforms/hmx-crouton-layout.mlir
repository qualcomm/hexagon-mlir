//===- hmx-crouton-layout.mlir - the crouton layout in linalg vocabulary --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The crouton layout is stock linalg vocabulary: packing a 32x32 fp16 tile has
// to split ONLY the row dimension, which yields exactly memref<16x32x2xf16>.
//
// This file is also a guard. Writing inner_tiles = [2, 1] looks equivalent but is
// rejected by linalg.pack ("packed rank != (unpacked rank + num tiling factors)"):
// a tile of 1 still adds a unit inner dimension, giving tensor<16x32x2x1>.
//
// RUN: linalg-hexagon-opt %s -fold-pack-unpack-constants -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @crouton_form
// CHECK: linalg.pack %{{.*}} inner_dims_pos = [0] inner_tiles = [2]
// CHECK-SAME: tensor<32x32xf16> -> tensor<16x32x2xf16>
func.func @crouton_form(%w: tensor<32x32xf16>) -> tensor<16x32x2xf16> {
  %e = tensor.empty() : tensor<16x32x2xf16>
  %p = linalg.pack %w inner_dims_pos = [0] inner_tiles = [2] into %e
       : tensor<32x32xf16> -> tensor<16x32x2xf16>
  return %p : tensor<16x32x2xf16>
}

// -----

// A constant weight is packed at compile time: the pack folds into a new
// constant, so the weight reaches the kernel already in crouton layout. This is
// the whole of step (1) of the pack-elimination ladder, for free.
// CHECK-LABEL: func.func @constant_weight_is_prepacked
// CHECK-NOT: linalg.pack
// CHECK: arith.constant dense<1.000000e+00> : tensor<16x32x2xf16>
func.func @constant_weight_is_prepacked() -> tensor<16x32x2xf16> {
  %w = arith.constant dense<1.000000e+00> : tensor<32x32xf16>
  %e = tensor.empty() : tensor<16x32x2xf16>
  %p = linalg.pack %w inner_dims_pos = [0] inner_tiles = [2] into %e
       : tensor<32x32xf16> -> tensor<16x32x2xf16>
  return %p : tensor<16x32x2xf16>
}

// -----

// ... and it scales to a real weight: 512x512 folds too.
// CHECK-LABEL: func.func @large_constant_weight_is_prepacked
// CHECK-NOT: linalg.pack
// CHECK: arith.constant dense<2.000000e+00> : tensor<256x512x2xf16>
func.func @large_constant_weight_is_prepacked() -> tensor<256x512x2xf16> {
  %w = arith.constant dense<2.000000e+00> : tensor<512x512xf16>
  %e = tensor.empty() : tensor<256x512x2xf16>
  %p = linalg.pack %w inner_dims_pos = [0] inner_tiles = [2] into %e
       : tensor<512x512xf16> -> tensor<256x512x2xf16>
  return %p : tensor<256x512x2xf16>
}
