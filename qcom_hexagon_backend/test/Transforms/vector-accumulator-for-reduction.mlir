//===- vector-accumulator-for-reduction.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s --hexagon-vector-accumulator-reduction | FileCheck %s
//
// Smoke test for VectorAccumulatorForReductionPass: confirms the pass
// fires on the canonical pattern per-tile scalar accumulator inside
// a tiled reduction.
//
// Pattern matched (per row):
//   scf.for %r = 0 to R step 1 iter_args(%acc = %init) -> tensor<Rxf32>
//     scf.for %c = 0 to C step S iter_args(%v = %acc) -> tensor<Rxf32>
//       %tile = extract_slice %input[%r, %c] [1, S] [1, 1]   : tensor<S>
//       %prev = extract_slice %v[%r] [1] [1]                  : tensor<f32>
//       %red  = linalg.generic {reduction} ins(%tile) outs(%prev) -> tensor<f32>
//       %nv   = insert_slice %red into %v[%r] [1] [1]
//       scf.yield %nv
//     scf.yield %inner
//
// After: the inner-loop iter-arg becomes a vector accumulator of width S;
// a single horizontal reduction is emitted once after the inner loop.

#map  = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

func.func @row_sum(%input: tensor<8x32xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0  : index
  %c1 = arith.constant 1  : index
  %c8 = arith.constant 8  : index
  %c32 = arith.constant 32 : index

  %res = scf.for %r = %c0 to %c8 step %c1
      iter_args(%acc = %init) -> (tensor<8xf32>) {
    %inner = scf.for %c = %c0 to %c32 step %c8
        iter_args(%v = %acc) -> (tensor<8xf32>) {
      %tile = tensor.extract_slice %input[%r, %c] [1, 8] [1, 1]
          : tensor<8x32xf32> to tensor<8xf32>
      %prev = tensor.extract_slice %v[%r] [1] [1]
          : tensor<8xf32> to tensor<f32>
      %red = linalg.generic
          {indexing_maps = [#map, #map1], iterator_types = ["reduction"]}
          ins(%tile : tensor<8xf32>) outs(%prev : tensor<f32>) {
        ^bb0(%a: f32, %o: f32):
          %s = arith.addf %a, %o : f32
          linalg.yield %s : f32
      } -> tensor<f32>
      %ins = tensor.insert_slice %red into %v[%r] [1] [1]
          : tensor<f32> into tensor<8xf32>
      scf.yield %ins : tensor<8xf32>
    }
    scf.yield %inner : tensor<8xf32>
  }
  return %res : tensor<8xf32>
}

// CHECK-LABEL: func.func @row_sum
//
// The matched pattern is rewritten: the inner loop now carries a width-S
// vector accumulator (tensor<8xf32>) instead of writing back a rank-0 scalar
// every iteration. After the inner loop, a single horizontal reduction is
// emitted, then inserted into the outer accumulator.
//
// CHECK:      scf.for {{.*}} step %c1 iter_args({{.*}}) -> (tensor<8xf32>)
//
// Inner loop's iter-arg type is the width-S vector accumulator.
// CHECK:        scf.for {{.*}} step %c8 iter_args({{.*}}) -> (tensor<8xf32>)
// CHECK:          arith.addf
// CHECK:          scf.yield {{.*}} : tensor<8xf32>
//
// Exactly one horizontal reduction (1-D -> rank-0) emitted after the inner
// loop, inserted back into the outer rank-1 accumulator.
// CHECK:        linalg.generic {{.*}}iterator_types = ["reduction"]
// CHECK-SAME:     ins({{.*}} : tensor<8xf32>) outs({{.*}} : tensor<f32>)
// CHECK:        tensor.insert_slice {{.*}} : tensor<f32> into tensor<8xf32>
// CHECK:        scf.yield {{.*}} : tensor<8xf32>
