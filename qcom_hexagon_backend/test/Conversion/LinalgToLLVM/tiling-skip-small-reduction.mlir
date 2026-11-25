//===- tiling-skip-small-reduction.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-tiling | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// Reduction from tensor<64xf32> to scalar should NOT be tiled.
// 64 x f32 = 256 bytes = one HVX vector pair. The Hexagon backend can
// lower this as a single shuffle-reduce tree.

// CHECK-LABEL: func.func @reduction_64xf32_to_scalar
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK-SAME:      ins(%{{.*}} : tensor<64xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<f32>)
func.func @reduction_64xf32_to_scalar(%input: tensor<64xf32>,
    %init: tensor<f32>) -> tensor<f32> {
  %result = linalg.generic {
      indexing_maps = [#map, #map1], iterator_types = ["reduction"]}
      ins(%input : tensor<64xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// -----

// Reduction from tensor<128xf16> to scalar should NOT be tiled.
// 128 x f16 = 256 bytes = one HVX vector pair.

#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @reduction_128xf16_to_scalar
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK-SAME:      ins(%{{.*}} : tensor<128xf16>)
func.func @reduction_128xf16_to_scalar(%input: tensor<128xf16>,
    %init: tensor<f16>) -> tensor<f16> {
  %result = linalg.generic {
      indexing_maps = [#map4, #map5], iterator_types = ["reduction"]}
      ins(%input : tensor<128xf16>) outs(%init : tensor<f16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %out : f16
      linalg.yield %add : f16
  } -> tensor<f16>
  return %result : tensor<f16>
}

// -----

// Large reduction from tensor<256xf32> to scalar SHOULD be tiled.
// 256 x f32 = 1024 bytes > 2 * HVX vector.
// Ideally, the split reduction pass should convert this loop into two step
// reduction, performing partial reduction into a vector of 32xf32
// followed by a vector->scalar reduction. So, the hexagon tiling pass
// shouldn't encounter a loop like this.

#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @reduction_256xf32_to_scalar_tiled
// CHECK:         scf.for
func.func @reduction_256xf32_to_scalar_tiled(%input: tensor<256xf32>,
    %init: tensor<f32>) -> tensor<f32> {
  %result = linalg.generic {
      indexing_maps = [#map6, #map7], iterator_types = ["reduction"]}
      ins(%input : tensor<256xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// -----

// Parallel op on tensor<64xf32> SHOULD be tiled (2x HVX vector, not a
// reduction). Only reductions skip tiling at vector-pair size.

#map8 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @parallel_64xf32_still_tiled
// CHECK:         scf.for
func.func @parallel_64xf32_still_tiled(%a: tensor<64xf32>,
    %b: tensor<64xf32>) -> tensor<64xf32> {
  %result = linalg.generic {
      indexing_maps = [#map8, #map8], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf32>) outs(%b : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}
