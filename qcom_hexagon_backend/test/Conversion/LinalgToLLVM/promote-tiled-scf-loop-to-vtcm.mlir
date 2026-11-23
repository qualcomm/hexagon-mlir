//===- promote-tiled-scf-loop-to-vtcm.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(promote-tiled-scf-loop-to-vtcm))' | FileCheck %s

// Block-softmax IR fed into PromoteTiledSCFLoopToVTCMPass. The input is
// the post-HexagonFusion form :
//   - one tiled DDR input  (%arg0 → reinterpret_cast → to_tensor)
//   - one tiled DDR output (%arg1 ← reinterpret_cast ← materialize_in_destination)
//   - body: max-reduction, exp(x - max), sum-reduction, divide
//   - loop bounds expressed in i32; per-iteration flat offset is
//     `index_cast(iv) * 512` matching the [32, 512] tile's row stride.
//
// Expected rewrite shape:
//   - Both reinterpret_casts hoisted to full <114688x512> views with iv=0
//     symbolically substituted into the offset (still emitted via the
//     index_cast/muli chain, evaluating to 0).
//   - Two bufferization.to_tensor hoisted (input restrict, output restrict
//     writable).
//   - The i32 loop bounds index_cast'd; the new scf.for runs over index, has
//     a tensor<114688x512xf32> iter-arg seeded from the output to_tensor,
//     yields a tensor<114688x512xf32>, and is tagged {all_parallel, tiled_generic}.
//   - Per-tile path inside the body: extract_slice from the full input, an
//     alloc_tensor()+copy in VTCM (memory_space = 1), the existing softmax
//     compute (max-reduction, subf+exp, sum-reduction, div+mul) preserved
//     verbatim, then a final alloc_tensor()+copy of the result (output
//     staging) and an insert_slice into the iter-arg.
//   - One materialize_in_destination after the loop.

// CHECK-LABEL: func.func @block_softmax_kernel
// CHECK-SAME:    (%[[ARG0:.*]]: memref<*xf32> {{.*}}, %[[ARG1:.*]]: memref<*xf32>
//
// Original i32 loop bounds become index via index_cast.
// CHECK:      %[[LB:[0-9a-zA-Z_]+]] = arith.index_cast %{{[0-9a-zA-Z_]+}} : i32 to index
// CHECK:      %[[UB:[0-9a-zA-Z_]+]] = arith.index_cast %c114688_i32 : i32 to index
// CHECK:      %[[STEP:[0-9a-zA-Z_]+]] = arith.index_cast %{{[0-9a-zA-Z_]+}} : i32 to index
//
// Both reinterpret_casts hoisted to full shape, dynamic offset (iv=0 after
// symbolic substitution).
// CHECK:      %[[RC_IN:[0-9a-zA-Z_]+]] = memref.reinterpret_cast %[[ARG0]]
// CHECK-SAME:   sizes: [114688, 512], strides: [512, 1]
// CHECK-SAME:   memref<*xf32> to memref<114688x512xf32, strided<[512, 1], offset: ?>>
// CHECK:      %[[RC_OUT:[0-9a-zA-Z_]+]] = memref.reinterpret_cast %[[ARG1]]
// CHECK-SAME:   sizes: [114688, 512], strides: [512, 1]
// CHECK-SAME:   memref<*xf32> to memref<114688x512xf32, strided<[512, 1], offset: ?>>
//
// Hoisted to_tensors (input is plain restrict, output is restrict writable).
// CHECK:      %[[T_IN:[0-9a-zA-Z_]+]] = bufferization.to_tensor %[[RC_IN]] restrict :
// CHECK-NOT:    writable
// CHECK-SAME:   tensor<114688x512xf32>
// CHECK:      %[[T_OUT:[0-9a-zA-Z_]+]] = bufferization.to_tensor %[[RC_OUT]] restrict writable
// CHECK-SAME:   tensor<114688x512xf32>
//
// New scf.for over the index domain, iter-arg seeded from the output to_tensor.
// CHECK:      %[[FOR:[0-9a-zA-Z_]+]] = scf.for %[[IV:[0-9a-zA-Z_]+]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:[0-9a-zA-Z_]+]] = %[[T_OUT]]) -> (tensor<114688x512xf32>) {
//
// Per-tile path: extract_slice from full input, then VTCM staging copy.
// CHECK:        %[[IN_SLICE:[0-9a-zA-Z_]+]] = tensor.extract_slice %[[T_IN]][%[[IV]], 0] [32, 512] [1, 1]
// CHECK-SAME:     tensor<114688x512xf32> to tensor<32x512xf32>
// CHECK:        %[[IN_VTCM:[0-9a-zA-Z_]+]] = bufferization.alloc_tensor() copy(%[[IN_SLICE]])
// CHECK-SAME:     {memory_space = 1 : i64} : tensor<32x512xf32>
//
// Original softmax compute preserved (max → subf+exp → sum → divf+mulf chain).
// CHECK:        linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[IN_VTCM]] : tensor<32x512xf32>)
// CHECK:          arith.maxnumf
// CHECK:        linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]
// CHECK:          arith.subf
// CHECK:          math.exp
// CHECK:        linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]
// CHECK:          arith.addf
// CHECK:        %[[OUT_TILE:[0-9a-zA-Z_]+]] = linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]
// CHECK:          arith.divf
// CHECK:          arith.mulf
//
// Final per-iteration output staging (alloc_tensor + copy in memory_space 1)
// then insert_slice back into the iter-arg.
// CHECK:        %[[OUT_VTCM:[0-9a-zA-Z_]+]] = bufferization.alloc_tensor() copy(%[[OUT_TILE]])
// CHECK-SAME:     {memory_space = 1 : i64} : tensor<32x512xf32>
// CHECK:        %[[INS:[0-9a-zA-Z_]+]] = tensor.insert_slice %[[OUT_VTCM]] into %[[ACC]][%[[IV]], 0] [32, 512] [1, 1]
// CHECK-SAME:     tensor<32x512xf32> into tensor<114688x512xf32>
// CHECK:        scf.yield %[[INS]] : tensor<114688x512xf32>
// CHECK:      } {all_parallel, tiled_generic}
//
// Single materialize after the loop, into the hoisted output rc.
// CHECK:      bufferization.materialize_in_destination %[[FOR]] in restrict writable %[[RC_OUT]]
// CHECK:      return

func.func @block_softmax_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant 1.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c114688_i32 = arith.constant 114688 : i32
  %c512 = arith.constant 512 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = arith.muli %arg6, %arg2 : i32
  %1 = arith.addi %0, %arg5 : i32
  %2 = arith.muli %arg2, %arg3 : i32
  %3 = arith.muli %1, %c32_i32 : i32
  %4 = arith.muli %2, %c32_i32 : i32
  scf.for %arg8 = %3 to %c114688_i32 step %4  : i32 {
    %5 = arith.index_cast %arg8 : i32 to index
    %6 = arith.muli %5, %c512 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [32, 512], strides: [512, 1] : memref<*xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [32, 512], strides: [512, 1] : memref<*xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %7 = bufferization.to_tensor %reinterpret_cast_2 restrict : memref<32x512xf32, strided<[512, 1], offset: ?>> to tensor<32x512xf32>
    %8 = tensor.empty() : tensor<32x512xf32>
    %9 = tensor.empty() : tensor<32xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<32x512xf32>) outs(%10 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.maxnumf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<32xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %11 : tensor<32x512xf32>, tensor<32xf32>) outs(%8 : tensor<32x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %16 = arith.subf %in, %in_3 : f32
      %17 = math.exp %16 : f32
      linalg.yield %17 : f32
    } -> tensor<32x512xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%12 : tensor<32x512xf32>) outs(%13 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.addf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<32xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %14 : tensor<32x512xf32>, tensor<32xf32>) outs(%8 : tensor<32x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %16 = arith.divf %cst, %in_3 : f32
      %17 = arith.mulf %in, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<32x512xf32>
    bufferization.materialize_in_destination %15 in restrict writable %reinterpret_cast : (tensor<32x512xf32>, memref<32x512xf32, strided<[512, 1], offset: ?>>) -> ()
  }
  return
}
