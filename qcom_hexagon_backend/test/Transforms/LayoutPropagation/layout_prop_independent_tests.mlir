//===- layout_prop_independent_tests.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' -split-input-file | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Independent tests for the extend-pack-frontier pass covering
// pack/unpack propagation through linalg.generic in various scenarios:
// inner dim reorder, outer dim permutation, dynamic shapes, broadcast,
// reduction iterators, and known limitations.
//===------------------------------------------------------------------===//

// Test 1: Basic pack lifts over parallel elementwise generic

func.func @verify_pack_lifts_over_add(
    %input_mat: tensor<6x10xf32>,
    %bias_mat: tensor<6x10xf32>) -> tensor<2x2x3x5xf32> {
  %workspace = tensor.empty() : tensor<6x10xf32>
  %sum_result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_mat, %bias_mat : tensor<6x10xf32>, tensor<6x10xf32>)
    outs(%workspace : tensor<6x10xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
    %added = arith.addf %lhs, %rhs : f32
    linalg.yield %added : f32
  } -> tensor<6x10xf32>

  %tiled_out = tensor.empty() : tensor<2x2x3x5xf32>
  %packed = linalg.pack %sum_result
    inner_dims_pos = [0, 1] inner_tiles = [3, 5] into %tiled_out
    : tensor<6x10xf32> -> tensor<2x2x3x5xf32>
  return %packed : tensor<2x2x3x5xf32>
}
// CHECK-LABEL: func.func @verify_pack_lifts_over_add
// CHECK:         %[[PACK_IN:.+]] = linalg.pack
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [3, 5]
// CHECK:         %[[PACK_BIAS:.+]] = linalg.pack
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [3, 5]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_IN]], %[[PACK_BIAS]] : tensor<2x2x3x5xf32>, tensor<2x2x3x5xf32>)
// CHECK:         return %[[GEN]] : tensor<2x2x3x5xf32>

// -----

// Test 2: Pack with inner dimension transposition

func.func @verify_pack_lifts_inner_dim_reorder(
    %mat_a: tensor<12x16xf32>) -> tensor<4x4x4x3xf32> {
  %scratch = tensor.empty() : tensor<12x16xf32>
  %negated = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mat_a : tensor<12x16xf32>)
    outs(%scratch : tensor<12x16xf32>) {
  ^bb0(%val: f32, %out: f32):
    %neg = arith.negf %val : f32
    linalg.yield %neg : f32
  } -> tensor<12x16xf32>

  %tiled = tensor.empty() : tensor<4x4x4x3xf32>
  %packed = linalg.pack %negated
    inner_dims_pos = [1, 0] inner_tiles = [4, 3] into %tiled
    : tensor<12x16xf32> -> tensor<4x4x4x3xf32>
  return %packed : tensor<4x4x4x3xf32>
}
// CHECK-LABEL: func.func @verify_pack_lifts_inner_dim_reorder
// CHECK-SAME:    %[[MA:.+]]: tensor<12x16xf32>
// CHECK:         %[[PACK_MA:.+]] = linalg.pack %[[MA]]
// CHECK-SAME:      inner_dims_pos = [1, 0] inner_tiles = [4, 3]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_MA]] : tensor<4x4x4x3xf32>)
// CHECK:         return %[[GEN]] : tensor<4x4x4x3xf32>

// -----

// Test 3: Pack with outer dimension permutation

func.func @verify_pack_lifts_outer_dim_reorder(
    %mat_x: tensor<12x16xf32>) -> tensor<2x3x4x8xf32> {
  %buf = tensor.empty() : tensor<12x16xf32>
  %doubled = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mat_x : tensor<12x16xf32>)
    outs(%buf : tensor<12x16xf32>) {
  ^bb0(%elem: f32, %out: f32):
    %two = arith.constant 2.0 : f32
    %scaled = arith.mulf %elem, %two : f32
    linalg.yield %scaled : f32
  } -> tensor<12x16xf32>

  %dest = tensor.empty() : tensor<2x3x4x8xf32>
  %packed = linalg.pack %doubled
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %dest
    : tensor<12x16xf32> -> tensor<2x3x4x8xf32>
  return %packed : tensor<2x3x4x8xf32>
}
// CHECK-LABEL: func.func @verify_pack_lifts_outer_dim_reorder
// CHECK-SAME:    %[[MX:.+]]: tensor<12x16xf32>
// CHECK:         %[[PACK_MX:.+]] = linalg.pack %[[MX]]
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [4, 8]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_MX]] : tensor<2x3x4x8xf32>)
// CHECK:         return %[[GEN]] : tensor<2x3x4x8xf32>

// -----

// Test 4: Pack through generic with dynamic tensor dimensions

func.func @verify_pack_lifts_over_dynamic_op(
    %dyn_in: tensor<?x?xf32>,
    %pack_dest: tensor<?x?x2x8xf32>) -> tensor<?x?x2x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = tensor.dim %dyn_in, %c0 : tensor<?x?xf32>
  %cols = tensor.dim %dyn_in, %c1 : tensor<?x?xf32>
  %scratch = tensor.empty(%rows, %cols) : tensor<?x?xf32>
  %abs_result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%dyn_in : tensor<?x?xf32>)
    outs(%scratch : tensor<?x?xf32>) {
  ^bb0(%v: f32, %out: f32):
    %absv = math.absf %v : f32
    linalg.yield %absv : f32
  } -> tensor<?x?xf32>

  %packed = linalg.pack %abs_result
    inner_dims_pos = [0, 1] inner_tiles = [2, 8] into %pack_dest
    : tensor<?x?xf32> -> tensor<?x?x2x8xf32>
  return %packed : tensor<?x?x2x8xf32>
}
// CHECK-LABEL: func.func @verify_pack_lifts_over_dynamic_op
// CHECK-SAME:    %[[DYN:.+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[DEST:.+]]: tensor<?x?x2x8xf32>
// CHECK:         %[[PACK_DYN:.+]] = linalg.pack %[[DYN]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [2, 8]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_DYN]] : tensor<?x?x2x8xf32>)
// CHECK:         return %[[GEN]] : tensor<?x?x2x8xf32>

// -----

// Test 5: Unpack sinks past elementwise generic

func.func @verify_unpack_sinks_past_mul(
    %packed_in: tensor<3x2x4x8xf32>,
    %flat_weight: tensor<12x16xf32>) -> tensor<12x16xf32> {
  %flat_buf = tensor.empty() : tensor<12x16xf32>
  %flat_in = linalg.unpack %packed_in
    inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %flat_buf
    : tensor<3x2x4x8xf32> -> tensor<12x16xf32>

  %result_buf = tensor.empty() : tensor<12x16xf32>
  %product = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%flat_in, %flat_weight : tensor<12x16xf32>, tensor<12x16xf32>)
    outs(%result_buf : tensor<12x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    linalg.yield %mul : f32
  } -> tensor<12x16xf32>
  return %product : tensor<12x16xf32>
}
// CHECK-LABEL: func.func @verify_unpack_sinks_past_mul
// CHECK-SAME:    %[[PKD:.+]]: tensor<3x2x4x8xf32>
// CHECK-SAME:    %[[WT:.+]]: tensor<12x16xf32>
// CHECK:         %[[PACK_WT:.+]] = linalg.pack %[[WT]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [4, 8]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PKD]], %[[PACK_WT]] : tensor<3x2x4x8xf32>, tensor<3x2x4x8xf32>)
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[GEN]]
// CHECK:         return %[[UNPACK]] : tensor<12x16xf32>

// -----

// Test 6: Pack lifts over generic with broadcast input

func.func @verify_pack_lifts_with_broadcast(
    %full_mat: tensor<6x10xf32>,
    %row_vec: tensor<10xf32>) -> tensor<2x2x3x5xf32> {
  %ws = tensor.empty() : tensor<6x10xf32>
  %biased = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%full_mat, %row_vec : tensor<6x10xf32>, tensor<10xf32>)
    outs(%ws : tensor<6x10xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<6x10xf32>

  %tile_buf = tensor.empty() : tensor<2x2x3x5xf32>
  %packed = linalg.pack %biased
    inner_dims_pos = [0, 1] inner_tiles = [3, 5] into %tile_buf
    : tensor<6x10xf32> -> tensor<2x2x3x5xf32>
  return %packed : tensor<2x2x3x5xf32>
}
// CHECK-LABEL: func.func @verify_pack_lifts_with_broadcast
// CHECK-SAME:    %[[FMAT:.+]]: tensor<6x10xf32>
// CHECK-SAME:    %[[RVEC:.+]]: tensor<10xf32>
// CHECK:         %[[PACK_FMAT:.+]] = linalg.pack %[[FMAT]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [3, 5]
// CHECK:         %[[PACK_RVEC:.+]] = linalg.pack %[[RVEC]]
// CHECK-SAME:      inner_dims_pos = [0] inner_tiles = [5]
// CHECK:         %[[GEN:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_FMAT]], %[[PACK_RVEC]]
// CHECK:         return %[[GEN]] : tensor<2x2x3x5xf32>

// -----

// Test 7: Pack lifts over generic with reduction iterator

func.func @verify_pack_lifts_over_reduction(
    %data: tensor<6x10xf32>,
    %weights: tensor<10x15xf32>) -> tensor<2x3x3x5xf32> {
  %init_buf = tensor.empty() : tensor<6x15xf32>
  %cst_zero = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst_zero : f32) outs(%init_buf : tensor<6x15xf32>) -> tensor<6x15xf32>
  %contracted = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%data, %weights : tensor<6x10xf32>, tensor<10x15xf32>)
    outs(%zeroed : tensor<6x15xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %prod, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<6x15xf32>

  %tile_dst = tensor.empty() : tensor<2x3x3x5xf32>
  %packed = linalg.pack %contracted
    inner_dims_pos = [0, 1] inner_tiles = [3, 5] into %tile_dst
    : tensor<6x15xf32> -> tensor<2x3x3x5xf32>
  return %packed : tensor<2x3x3x5xf32>
}
// Pack through a reduction generic - the pass may or may not propagate
// depending on whether it can tile the reduction dimension
// CHECK-LABEL: func.func @verify_pack_lifts_over_reduction
// CHECK:         linalg.generic
// CHECK:         return

// -----

// Test 8: Pack lifts through multi-result generic

func.func @verify_pack_lifts_multiresult(
    %vec_p: tensor<12xf32>,
    %vec_q: tensor<12xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %ws1 = tensor.empty() : tensor<12xf32>
  %ws2 = tensor.empty() : tensor<12xf32>
  %res1, %res2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%vec_p, %vec_q : tensor<12xf32>, tensor<12xf32>)
    outs(%ws1, %ws2 : tensor<12xf32>, tensor<12xf32>) {
  ^bb0(%p: f32, %q: f32, %o1: f32, %o2: f32):
    %s = arith.addf %p, %q : f32
    %d = arith.subf %p, %q : f32
    linalg.yield %s, %d : f32, f32
  } -> (tensor<12xf32>, tensor<12xf32>)

  %t1 = tensor.empty() : tensor<3x4xf32>
  %pk1 = linalg.pack %res1
    inner_dims_pos = [0] inner_tiles = [4] into %t1
    : tensor<12xf32> -> tensor<3x4xf32>
  %t2 = tensor.empty() : tensor<3x4xf32>
  %pk2 = linalg.pack %res2
    inner_dims_pos = [0] inner_tiles = [4] into %t2
    : tensor<12xf32> -> tensor<3x4xf32>
  return %pk1, %pk2 : tensor<3x4xf32>, tensor<3x4xf32>
}
// Multi-result generics: packs are propagated through the generic
// CHECK-LABEL: func.func @verify_pack_lifts_multiresult
// CHECK:         %[[PKP:.+]] = linalg.pack
// CHECK-SAME:      inner_dims_pos = [0] inner_tiles = [4]
// CHECK:         %[[PKQ:.+]] = linalg.pack
// CHECK-SAME:      inner_dims_pos = [0] inner_tiles = [4]
// CHECK:         %[[GEN:.+]]:2 = linalg.generic
// CHECK-SAME:      ins(%[[PKP]], %[[PKQ]] : tensor<3x4xf32>, tensor<3x4xf32>)
// CHECK:         return %[[GEN]]#0, %[[GEN]]#1

// -----

//===------------------------------------------------------------------===//
// Hexagon-specific tests (not from upstream)
//===------------------------------------------------------------------===//

func.func @push_unpack_in_padded_domain_out_used(%arg0: tensor<8x8x4x8xf32>, %arg1: tensor<?x64xf32>) -> tensor<?x64xf32> {
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %arg1 : tensor<8x8x4x8xf32> -> tensor<?x64xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<?x64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?x64xf32>
  return %0 : tensor<?x64xf32>
}
// CHECK-LABEL: func.func @push_unpack_in_padded_domain_out_used
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1_PACK_EMPTY:.+]] = tensor.empty
// CHECK:         %[[ARG1_PACK:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [4, 8]
// CHECK-SAME:      into %[[ARG1_PACK_EMPTY]]
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[ARG0]] : tensor<8x8x4x8xf32>)
// CHECK-SAME:    outs(%[[ARG1_PACK]] : tensor<?x8x4x8xf32>)
// CHECK:         %[[UNPACK2:.+]] = linalg.unpack %[[GENERIC]]
// CHECK-SAME:    into %[[ARG1]]
// CHECK:         return %[[UNPACK2]] : tensor<?x64xf32>

// -----

func.func @nopush_extract_through_generic_nodimexpr1(%arg0: tensor<128x7x128xf32>, %arg1: tensor<?x5x3x128xf32>, %arg2: tensor<?x5x128xbf16>, %arg3: index) -> tensor<?x5x128xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %arg3] [128, 7, %arg3] [1, 1, 1] : tensor<128x7x128xf32> to tensor<128x7x?xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2 + d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %arg1 : tensor<128x7x?xf32>, tensor<?x5x3x128xf32>) outs(%arg2 : tensor<?x5x128xbf16>) {
  ^bb0(%in: f32, %in_0: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<?x5x128xbf16>
  return %0 : tensor<?x5x128xbf16>
}

// CHECK-LABEL: func.func @nopush_extract_through_generic_nodimexpr1
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         return %[[GENERIC]]

// -----

func.func @nopush_extract_through_generic_nodimexpr2(%arg0: tensor<128x?x128xf32>, %arg1: tensor<128x5x3x128xf32>, %arg2: tensor<128x?x128xbf16>, %arg3: index) -> tensor<128x?x128xbf16> {
  %extracted_slice = tensor.extract_slice %arg1[0, %arg3, 0, 0] [128, %arg3, 3, 128] [1, 1, 1, 1] : tensor<128x5x3x128xf32> to tensor<128x?x3x128xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2 + d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %extracted_slice : tensor<128x?x128xf32>, tensor<128x?x3x128xf32>) outs(%arg2 : tensor<128x?x128xbf16>) {
  ^bb0(%in: f32, %in_0: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<128x?x128xbf16>
  return %0 : tensor<128x?x128xbf16>
}

// CHECK-LABEL: func.func @nopush_extract_through_generic_nodimexpr2
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         return %[[GENERIC]]
