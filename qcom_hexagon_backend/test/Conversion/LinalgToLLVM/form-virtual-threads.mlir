//===- form-virtual-threads.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -form-virtual-threads="block-size=8" | FileCheck %s
// RUN: linalg-hexagon-opt %s -form-virtual-threads="preferred-num-threads=8" | FileCheck %s --check-prefix=SELECT
// RUN: not linalg-hexagon-opt %s -form-virtual-threads="preferred-num-threads=0" 2>&1 | FileCheck %s --check-prefix=INVALID-NUM-THREADS

#map = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map01 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map02 = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @kernel(%arg0: memref<1024x2048xf32>, %arg1: memref<1024x2048xf32>, %arg2: memref<1024x2048xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %3 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%0, %1 : tensor<1024x2048xf32>, tensor<1024x2048xf32>) outs(%2 : tensor<1024x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x2048xf32>
  bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<1024x2048xf32>, memref<1024x2048xf32>) -> ()
  return
}

// CHECK-LABEL: @kernel
// CHECK-SAME:  (%[[ARG0:.*]]: memref<1024x2048xf32>, %[[ARG1:.*]]: memref<1024x2048xf32>, %[[ARG2:.*]]: memref<1024x2048xf32>)
// CHECK:       %[[T0:.*]] = bufferization.to_tensor %[[ARG0]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[T1:.*]] = bufferization.to_tensor %[[ARG1]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[T2:.*]] = bufferization.to_tensor %[[ARG2]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[FORALL:.*]] = scf.forall (%[[IV:.*]]) = (0) to (1024) step (8) shared_outs(%[[SHARED_OUT:.*]] = %[[T2]]) -> (tensor<1024x2048xf32>) {
// CHECK:         %[[SLICE0:.*]] = tensor.extract_slice %[[T0]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[SLICE1:.*]] = tensor.extract_slice %[[T1]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[SLICE_OUT:.*]] = tensor.extract_slice %[[SHARED_OUT]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:        ins(%[[SLICE0]], %[[SLICE1]] : tensor<8x2048xf32>, tensor<8x2048xf32>) outs(%[[SLICE_OUT]] : tensor<8x2048xf32>) {
// CHECK:         ^bb0(%[[IN:.*]]: f32, %[[IN2:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:           %[[ADD:.*]] = arith.addf %[[IN]], %[[IN2]] : f32
// CHECK:           linalg.yield %[[ADD]] : f32
// CHECK:         } -> tensor<8x2048xf32>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[GENERIC]] into %[[SHARED_OUT]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<8x2048xf32> into tensor<1024x2048xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       bufferization.materialize_in_destination %[[FORALL]] in writable %[[ARG2]] : (tensor<1024x2048xf32>, memref<1024x2048xf32>) -> ()

func.func @prefer_saturating_dimension(%arg0: memref<2x65536xf32>, %arg1: memref<2x65536xf32>, %arg2: memref<2x65536xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x65536xf32> to tensor<2x65536xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x65536xf32> to tensor<2x65536xf32>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<2x65536xf32> to tensor<2x65536xf32>
  %3 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%0, %1 : tensor<2x65536xf32>, tensor<2x65536xf32>) outs(%2 : tensor<2x65536xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x65536xf32>
  bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<2x65536xf32>, memref<2x65536xf32>) -> ()
  return
}

func.func @prefer_largest_dimension(%arg0: memref<2x8192x4xf32>, %arg1: memref<2x4xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x8192x4xf32> to tensor<2x8192x4xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x4xf32> to tensor<2x4xf32>
  %2 = linalg.generic
         {indexing_maps = [#map3, #map02], iterator_types = ["parallel", "reduction", "parallel"]}
         ins(%0 : tensor<2x8192x4xf32>) outs(%1 : tensor<2x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<2x4xf32>
  bufferization.materialize_in_destination %2 in writable %arg1 : (tensor<2x4xf32>, memref<2x4xf32>) -> ()
  return
}

// The two small parallel dimensions could be flattened into eight tasks, but
// the single-dimension policy intentionally uses the largest one and creates
// four tasks instead.
func.func @prefer_largest_contiguous_dimension(%arg0: memref<2x4x8192xf32>, %arg1: memref<2x4xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x4x8192xf32> to tensor<2x4x8192xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x4xf32> to tensor<2x4xf32>
  %2 = linalg.generic
         {indexing_maps = [#map3, #map01], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%0 : tensor<2x4x8192xf32>) outs(%1 : tensor<2x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<2x4xf32>
  bufferization.materialize_in_destination %2 in writable %arg1 : (tensor<2x4xf32>, memref<2x4xf32>) -> ()
  return
}

func.func @tile_partial_last_tile(%arg0: memref<2x65537xf32>, %arg1: memref<2x65537xf32>, %arg2: memref<2x65537xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x65537xf32> to tensor<2x65537xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x65537xf32> to tensor<2x65537xf32>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<2x65537xf32> to tensor<2x65537xf32>
  %3 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%0, %1 : tensor<2x65537xf32>, tensor<2x65537xf32>) outs(%2 : tensor<2x65537xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x65537xf32>
  bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<2x65537xf32>, memref<2x65537xf32>) -> ()
  return
}

// SELECT-LABEL: @prefer_saturating_dimension
// SELECT:       scf.forall (%[[SATURATING_IV:.*]]) = (0) to (65536) step (8192)
// SELECT:       tensor.extract_slice %{{.*}}[0, %[[SATURATING_IV]]] [2, 8192] [1, 1] : tensor<2x65536xf32> to tensor<2x8192xf32>

// SELECT-LABEL: @prefer_largest_dimension
// SELECT:       scf.forall (%[[LARGEST_IV:.*]]) in (4)
// SELECT:       tensor.extract_slice %{{.*}}[0, 0, %[[LARGEST_IV]]] [2, 8192, 1] [1, 1, 1] : tensor<2x8192x4xf32> to tensor<2x8192x1xf32>

// SELECT-LABEL: @prefer_largest_contiguous_dimension
// SELECT:       scf.forall (%[[CONTIGUOUS_IV:.*]]) in (4)
// SELECT:       tensor.extract_slice %{{.*}}[0, %[[CONTIGUOUS_IV]], 0] [2, 1, 8192] [1, 1, 1] : tensor<2x4x8192xf32> to tensor<2x1x8192xf32>

// SELECT-LABEL: @tile_partial_last_tile
// SELECT:       scf.forall (%[[TAIL_IV:.*]]) = (0) to (65537) step (8192)
// SELECT:       %[[TAIL_SIZE:.*]] = affine.min #{{.*}}(%[[TAIL_IV]])
// SELECT:       tensor.extract_slice %{{.*}}[0, %[[TAIL_IV]]] [2, %[[TAIL_SIZE]]] [1, 1] : tensor<2x65537xf32> to tensor<2x?xf32>
// SELECT:       tensor.parallel_insert_slice %{{.*}} into %{{.*}}[0, %[[TAIL_IV]]] [2, %[[TAIL_SIZE]]] [1, 1] : tensor<2x?xf32> into tensor<2x65537xf32>

// INVALID-NUM-THREADS: error: preferred-num-threads must be greater than zero
