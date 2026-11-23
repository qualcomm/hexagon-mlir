//===- split-reduce-generic.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt --split-reduce-generic -cse -canonicalize %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @test_maxnumf_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2048xf32>) -> tensor<f32>
// CHECK-DAG: %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[CST]] into %[[ALLOC]][] : tensor<f32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [64, 32] : tensor<2048xf32> into tensor<64x32xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[PARTIAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["reduction", "parallel"]} ins(%[[EXPANDED]] : tensor<64x32xf32>) outs(%[[FILLED]] : tensor<32xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[MAX1:.*]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK: linalg.yield %[[MAX1]] : f32
// CHECK: } -> tensor<32xf32>
// CHECK: %[[FINAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["reduction"]} ins(%[[PARTIAL]] : tensor<32xf32>) outs(%[[INSERTED]] : tensor<f32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[MAX2:.*]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK: linalg.yield %[[MAX2]] : f32
// CHECK: } -> tensor<f32>
// CHECK: return %[[FINAL]] : tensor<f32>
func.func @test_maxnumf_reduction(%arg0: tensor<2048xf32>) -> tensor<f32> {
  %cst = arith.constant 0xFF800000 : f32
  %3 = bufferization.alloc_tensor() : tensor<f32>
  %inserted = tensor.insert %cst into %3[] : tensor<f32>

  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<2048xf32>) outs(%inserted : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.maxnumf %in, %out : f32
    linalg.yield %12 : f32
  } -> tensor<f32>

  return %4 : tensor<f32>
}

// Attention-style row reduction: preserve the 2048 parallel rows while
// splitting the 512-element reduction dimension.
#row_reduce_map = affine_map<(d0, d1) -> (d0, d1)>
#row_output_map = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @test_row_maxnumf_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2048x512xf32>) -> tensor<2048xf32>
// CHECK-DAG: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]{{\]}} output_shape [2048, 16, 32] : tensor<2048x512xf32> into tensor<2048x16x32xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2048x32xf32>
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[NEG_INF]] : f32) outs(%[[EMPTY]] : tensor<2048x32xf32>) -> tensor<2048x32xf32>
// CHECK: %[[PARTIAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[EXPANDED]] : tensor<2048x16x32xf32>) outs(%[[FILLED]] : tensor<2048x32xf32>)
// CHECK: %[[FINAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "reduction"]} ins(%[[PARTIAL]] : tensor<2048x32xf32>) outs(%{{.*}} : tensor<2048xf32>)
// CHECK: return %[[FINAL]] : tensor<2048xf32>
func.func @test_row_maxnumf_reduction(%arg0: tensor<2048x512xf32>)
    -> tensor<2048xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxnumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// The split-reduction pass runs after bufferization in some pipelines. Recover
// the fill initializer when it is wrapped in alloc_tensor copy(...).
// CHECK-LABEL: func.func @test_row_maxnumf_with_copied_fill
// CHECK-DAG: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[COPIED:.*]] = bufferization.alloc_tensor() copy(%{{.*}}) : tensor<2048xf32>
// CHECK: linalg.fill ins(%[[NEG_INF]] : f32) outs(%{{.*}} : tensor<2048x32xf32>)
func.func @test_row_maxnumf_with_copied_fill(%arg0: tensor<2048x512xf32>)
    -> tensor<2048xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %copied = bufferization.alloc_tensor() copy(%init) : tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%copied : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxnumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// CHECK-LABEL: func.func @test_row_maxnumf_with_output_arg
// CHECK: tensor.expand_shape %[[ARG:.*]] {{\[}}[0], [1, 2]{{\]}} output_shape [2048, 16, 32]
func.func @test_row_maxnumf_with_output_arg(
    %arg0: tensor<2048x512xf32>, %out: tensor<2048xf32>) -> tensor<2048xf32> {
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%out : tensor<2048xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %1 = arith.maxnumf %in, %acc : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// CHECK-LABEL: func.func @test_row_addf_reduction
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]{{\]}} output_shape [2048, 16, 32]
// CHECK: linalg.fill ins(%[[ZERO]] : f32) outs(%{{.*}} : tensor<2048x32xf32>)
func.func @test_row_addf_reduction(%arg0: tensor<2048x512xf32>)
    -> tensor<2048xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// Do not split outer (column) reductions; this pass targets the innermost
// reduction loop.
#column_reduce_map = affine_map<(d0, d1) -> (d0, d1)>
#column_output_map = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @test_column_addf_reduction_not_split
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic {{.*}}iterator_types = ["reduction", "parallel"]
// CHECK: arith.addf
func.func @test_column_addf_reduction_not_split(%arg0: tensor<512x2048xf32>)
    -> tensor<2048xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#column_reduce_map, #column_output_map],
    iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<512x2048xf32>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// RMSNorm and LayerNorm promote f16 elements before accumulating into f32.
// CHECK-LABEL: func.func @test_row_extf_addf_reduction
// f16 inputs use the 64-element data tile, so 512 splits into 8 x 64.
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]{{\]}} output_shape [2048, 8, 64]
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: arith.extf
// CHECK: arith.addf
func.func @test_row_extf_addf_reduction(%arg0: tensor<2048x512xf16>)
    -> tensor<2048xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf16>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f16, %out: f32):
      %1 = arith.extf %in : f16 to f32
      %2 = arith.addf %1, %out : f32
      linalg.yield %2 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// Keep the generic split-reduction coverage aligned with the floating-point
// combiners accepted by the tiling-pass split-reduction candidate check.
// CHECK-LABEL: func.func @test_row_minnumf_reduction
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<2048x32xf32>)
// CHECK: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: arith.minnumf
// CHECK: iterator_types = ["parallel", "reduction"]
func.func @test_row_minnumf_reduction(%arg0: tensor<2048x512xf32>)
    -> tensor<2048xf32> {
  %pos_inf = arith.constant 0x7F800000 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%pos_inf : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.minnumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}

// CHECK-LABEL: func.func @test_row_mulf_reduction
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<2048x32xf32>)
// CHECK: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: arith.mulf
// CHECK: iterator_types = ["parallel", "reduction"]
func.func @test_row_mulf_reduction(%arg0: tensor<2048x512xf32>)
    -> tensor<2048xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %empty = tensor.empty() : tensor<2048xf32>
  %init = linalg.fill ins(%one : f32) outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>
  %0 = linalg.generic {
    indexing_maps = [#row_reduce_map, #row_output_map],
    iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<2048xf32>
  return %0 : tensor<2048xf32>
}
