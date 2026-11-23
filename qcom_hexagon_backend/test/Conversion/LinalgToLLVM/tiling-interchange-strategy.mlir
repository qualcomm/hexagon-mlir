//===- tiling-interchange-strategy.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-tiling="use-interchange-vector=true" | FileCheck %s --check-prefix=ON
// RUN: linalg-hexagon-opt %s -hexagon-tiling="use-interchange-vector=false" | FileCheck %s --check-prefix=OFF

// Interchange applied when enabled.
// Inner dim = 8 f32 < dataTileSize = 32, outer dim = 64 >= 32.
// Interchange swaps the outer (64) to innermost and tiles by 32.
// When disabled, tiling bails out and the op is unchanged.

#map = affine_map<(d0, d1) -> (d0, d1)>

// ON-LABEL: func.func @interchange_applied
// ON:         scf.for
// ON:         tensor.extract_slice {{.*}} [32, 1] [1, 1]
// ON:         linalg.generic {{.*}} ins({{.*}} : tensor<32x1xf32>)

// OFF-LABEL: func.func @interchange_applied
// OFF-NOT:     scf.for
// OFF:         linalg.generic {{.*}} ins({{.*}} : tensor<64x8xf32>)
func.func @interchange_applied(%a: tensor<64x8xf32>, %b: tensor<64x8xf32>) -> tensor<64x8xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<64x8xf32>) outs(%b : tensor<64x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<64x8xf32>
  return %0 : tensor<64x8xf32>
}
