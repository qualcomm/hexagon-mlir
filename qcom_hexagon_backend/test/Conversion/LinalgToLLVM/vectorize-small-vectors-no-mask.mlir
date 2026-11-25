//===- vectorize-small-vectors-no-mask.mlir -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s -hexagon-vectorization | FileCheck %s

// These tests exercise the "rely on the LLVM Hexagon backend" route for small
// (sub-HVX) vectors: the inner loop dimension is vectorized at its NATURAL
// width (no padding, no vector.mask). The backend is responsible for widening
// the sub-HVX vector and predicating partial loads/stores.

// Test 1: Small elementwise vector (no masking).
// Inner dim = 32 f16 elements = 64 bytes < 128 bytes (one HVX register).
// dataTileSize = 128/2 = 64. Since 32 < 64 the loop is still vectorized, but
// at its natural width vector<32xf16> and WITHOUT any vector.mask.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @small_elemwise_f16
// CHECK-NOT:     vector.mask
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.transfer_read {{.*}} : tensor<32xf16>, vector<32xf16>
// CHECK:         arith.addf {{.*}} : vector<32xf16>
// CHECK:         vector.transfer_write {{.*}} : vector<32xf16>, tensor<32xf16>
func.func @small_elemwise_f16(%a: tensor<32xf16>, %b: tensor<32xf16>) -> tensor<32xf16> {
  %result = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<32xf16>) outs(%b : tensor<32xf16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %out : f16
      linalg.yield %add : f16
  } -> tensor<32xf16>
  return %result : tensor<32xf16>
}

// -----

// Test 2: Small reduction (no masking).
// Inner dim = 32 f16 elements = 64 bytes < 128 bytes. dataTileSize = 64.
// The reduction is vectorized at natural width vector<32xf16> and produces a
// plain (unmasked) vector.multi_reduction; the backend lowers it to a
// valign+vadd tree once fast-math flags are present. No vector.mask appears.

#map_red = affine_map<(d0) -> (d0)>
#map_acc = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @small_reduce_f16
// CHECK-NOT:     vector.mask
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.transfer_read {{.*}} : tensor<32xf16>, vector<32xf16>
// CHECK:         vector.multi_reduction <add>, {{.*}} : vector<32xf16> to f16
func.func @small_reduce_f16(%x: tensor<32xf16>, %init: tensor<f16>) -> tensor<f16> {
  %result = linalg.generic {
      indexing_maps = [#map_red, #map_acc], iterator_types = ["reduction"]}
      ins(%x : tensor<32xf16>) outs(%init : tensor<f16>) {
    ^bb0(%in: f16, %acc: f16):
      %add = arith.addf %in, %acc : f16
      linalg.yield %add : f16
  } -> tensor<f16>
  return %result : tensor<f16>
}

// -----

// Test 3: Perfect vector (no masking, unchanged behavior).
// Inner dim = 64 f16 elements = 128 bytes = exactly one HVX register.
// dataTileSize = 64, so innerLoopDim == dataTileSize: vectorized at width 64
// with no mask, exactly as before this change.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @perfect_vector_f16
// CHECK-NOT:     vector.mask
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.transfer_read {{.*}} : tensor<64xf16>, vector<64xf16>
// CHECK:         vector.transfer_write {{.*}} : vector<64xf16>, tensor<64xf16>
func.func @perfect_vector_f16(%a: tensor<64xf16>, %b: tensor<64xf16>) -> tensor<64xf16> {
  %result = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%b : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %out : f16
      linalg.yield %add : f16
  } -> tensor<64xf16>
  return %result : tensor<64xf16>
}
