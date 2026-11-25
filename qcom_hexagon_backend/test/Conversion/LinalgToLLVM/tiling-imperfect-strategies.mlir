//===- tiling-imperfect-strategies.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s -hexagon-tiling="split-tiling-range=false" | FileCheck %s --check-prefix=PAD
// RUN: linalg-hexagon-opt -split-input-file %s -hexagon-tiling="split-tiling-range=true" | FileCheck %s --check-prefix=SPLIT

// Test 1: Imperfect reduction (f32, C=112, dataTileSize=32) is NOT padded.
// 112 > 2*32 so the small-reduction bail-out does not fire.
// Inner iterator is reduction, so padding (which uses zero) is unsafe.
// With split-tiling-range=false: tiling bails, op unchanged.
// With split-tiling-range=true: split into head(96) + tail(16), head tiled.

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>

// PAD-LABEL: func.func @imperfect_reduction_112
// PAD-NOT:     tensor.pad
// PAD:         linalg.generic {{.*}} ins({{.*}} : tensor<112xf32>)

// SPLIT-LABEL: func.func @imperfect_reduction_112
// SPLIT:         tensor.extract_slice {{.*}} [96] [1] : tensor<112xf32>
// SPLIT:         scf.for
// SPLIT:         linalg.generic {{.*}} ins({{.*}} : tensor<32xf32>)
// SPLIT:         tensor.extract_slice {{.*}} [16] [1] : tensor<112xf32>
// SPLIT:         linalg.generic {{.*}} ins({{.*}} : tensor<16xf32>)
func.func @imperfect_reduction_112(%x: tensor<112xf32>, %init: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]}
       ins(%x : tensor<112xf32>) outs(%init : tensor<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %add = arith.addf %in, %acc : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Test 2: Imperfect parallel (f32, 8x40, inner=40, dataTileSize=32) IS padded.
// Inner iterator is parallel so zero-padding is safe.
// Outer is reduction -- verifies [numLoops-1] indexing (not [0]).
// With split-tiling-range=false: should pad inner from 40 to 64.
// With split-tiling-range=true: should split into head(32) + tail(8).

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>

// PAD-LABEL: func.func @imperfect_parallel_inner_padded
// PAD:         tensor.pad {{.*}} high[0, 24]
// PAD:         scf.for

// SPLIT-LABEL: func.func @imperfect_parallel_inner_padded
// SPLIT:         tensor.extract_slice {{.*}} [32] [1]
// SPLIT:         scf.for
func.func @imperfect_parallel_inner_padded(%x: tensor<8x40xf32>, %out: tensor<40xf32>) -> tensor<40xf32> {
  %r = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction", "parallel"]}
       ins(%x : tensor<8x40xf32>) outs(%out : tensor<40xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %add = arith.addf %in, %acc : f32
    linalg.yield %add : f32
  } -> tensor<40xf32>
  return %r : tensor<40xf32>
}

// -----

// Test 3: Perfect multi-loop tiling (8x64 f32, inner=64=2*dataTileSize).
// Tiled with tile sizes [1, 32], producing scf.for over inner with step 32.

#map5 = affine_map<(d0, d1) -> (d0, d1)>

// PAD-LABEL: func.func @perfect_multiloop
// PAD:         scf.for
// PAD:         tensor.extract_slice {{.*}} [1, 32] [1, 1]
// PAD:         linalg.generic {{.*}} ins({{.*}} : tensor<1x32xf32>)

// SPLIT-LABEL: func.func @perfect_multiloop
// SPLIT:         scf.for
// SPLIT:         tensor.extract_slice {{.*}} [1, 32] [1, 1]
// SPLIT:         linalg.generic {{.*}} ins({{.*}} : tensor<1x32xf32>)
func.func @perfect_multiloop(%a: tensor<8x64xf32>, %b: tensor<8x64xf32>) -> tensor<8x64xf32> {
  %0 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<8x64xf32>) outs(%b : tensor<8x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<8x64xf32>
  return %0 : tensor<8x64xf32>
}

// -----

// Test 4: Sub-HVX single-loop reduction stays un-tiled (bail-out).
// tensor<32xf16> -> f16, dataTileSize = 64, innerLoopRange=32 <= 64.

#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0) -> ()>

// PAD-LABEL: func.func @sub_hvx_reduction_untiled
// PAD-NOT:     scf.for
// PAD:         linalg.generic {{.*}} ins({{.*}} : tensor<32xf16>)

// SPLIT-LABEL: func.func @sub_hvx_reduction_untiled
// SPLIT-NOT:     scf.for
// SPLIT:         linalg.generic {{.*}} ins({{.*}} : tensor<32xf16>)
func.func @sub_hvx_reduction_untiled(%x: tensor<32xf16>, %init: tensor<f16>) -> tensor<f16> {
  %r = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["reduction"]}
       ins(%x : tensor<32xf16>) outs(%init : tensor<f16>) {
  ^bb0(%in: f16, %acc: f16):
    %add = arith.addf %in, %acc : f16
    linalg.yield %add : f16
  } -> tensor<f16>
  return %r : tensor<f16>
}
