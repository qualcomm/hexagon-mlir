//===- softmax-optimization.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-to-composite))' | FileCheck %s

// Test that the recognition pass replaces the softmax
// pattern with a single composite.softmax op.

// Softmax pattern (post-CSE, before fusion): 7 separate ops.
//   max-reduce -> broadcast_max -> sub -> exp -> sum-reduce ->
//   reciprocal -> broadcast_rcp -> mul

// CHECK-LABEL: func.func @softmax
// CHECK:         %[[INPUT:.*]] = bufferization.to_tensor
// CHECK:         composite.softmax axis = 1 %[[INPUT]], %{{.*}} : tensor<256x512xf32>, tensor<256x512xf32> -> tensor<256x512xf32>
// CHECK:         bufferization.materialize_in_destination
// CHECK-NOT:     arith.maxnumf
// CHECK-NOT:     math.exp
// CHECK-NOT:     arith.addf

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

func.func @softmax(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant 1.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c512 = arith.constant 512 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = arith.muli %arg5, %c256_i32 : i32
  %1 = arith.muli %arg2, %c256_i32 : i32
  scf.for %arg8 = %0 to %c1024_i32 step %1  : i32 {
    %2 = arith.index_cast %arg8 : i32 to index
    %3 = arith.muli %2, %c512 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [256, 512], strides: [512, 1] : memref<*xf32> to memref<256x512xf32, strided<[512, 1], offset: ?>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [256, 512], strides: [512, 1] : memref<*xf32> to memref<256x512xf32, strided<[512, 1], offset: ?>>
    %4 = bufferization.to_tensor %reinterpret_cast_2 restrict : memref<256x512xf32, strided<[512, 1], offset: ?>> to tensor<256x512xf32>
    %5 = tensor.empty() : tensor<256x512xf32>
    %6 = tensor.empty() : tensor<256xf32>
    // max-reduce
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<256xf32>) -> tensor<256xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<256x512xf32>) outs(%7 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.maxnumf %in, %out : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    // broadcast max
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<256xf32>) outs(%5 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x512xf32>
    // sub
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %9 : tensor<256x512xf32>, tensor<256x512xf32>) outs(%5 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %17 = arith.subf %in, %in_3 : f32
      linalg.yield %17 : f32
    } -> tensor<256x512xf32>
    // exp
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<256x512xf32>) outs(%5 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = math.exp %in : f32
      linalg.yield %17 : f32
    } -> tensor<256x512xf32>
    // sum-reduce
    %12 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<256xf32>) -> tensor<256xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%11 : tensor<256x512xf32>) outs(%12 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.addf %in, %out : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    // reciprocal (1/sum)
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%13 : tensor<256xf32>) outs(%6 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.divf %cst, %in : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    // broadcast reciprocal
    %15 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<256xf32>) outs(%5 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x512xf32>
    // final mul: exp * (1/sum)
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %15 : tensor<256x512xf32>, tensor<256x512xf32>) outs(%5 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %17 = arith.mulf %in, %in_3 : f32
      linalg.yield %17 : f32
    } -> tensor<256x512xf32>
    bufferization.materialize_in_destination %16 in restrict writable %reinterpret_cast : (tensor<256x512xf32>, memref<256x512xf32, strided<[512, 1], offset: ?>>) -> ()
  }
  return
}
