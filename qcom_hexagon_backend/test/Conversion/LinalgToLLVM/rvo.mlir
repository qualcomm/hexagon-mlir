//===- rvo.mlir -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(hexagon-rvo,canonicalize,cse,eliminate-empty-tensors),one-shot-bufferize,canonicalize)' \
// RUN:     | FileCheck %s -check-prefixes=CHECK

func.func @const_offset(%X: memref<*xf32>, %Y: memref<*xf32>, %Z: memref<*xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %X to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
    %0 = bufferization.to_tensor %reinterpret_cast restrict : memref<128xf32, strided<[1]>> to tensor<128xf32>

    %reinterpret_cast_0 = memref.reinterpret_cast %Y to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
    %1 = bufferization.to_tensor %reinterpret_cast_0 restrict : memref<128xf32, strided<[1]>> to tensor<128xf32>

    %2 = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
                        iterator_types = ["parallel"]}
                        ins(%0, %1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<128xf32>

    %reinterpret_cast_1 = memref.reinterpret_cast %Z to offset: [0], sizes: [128], strides: [1]
                                       : memref<*xf32> to memref<128xf32, strided<[1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_1
                                       : (tensor<128xf32>, memref<128xf32, strided<[1]>>) -> ()
    return
  }

// CHECK-LABEL: @const_offset
// CHECK: %[[X:.*]]: memref<*xf32>, %[[Y:.*]]: memref<*xf32>, %[[Z:.*]]: memref<*xf32>
//
// CHECK: %[[CAST:.+]] = memref.reinterpret_cast %[[Z]] to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
// CHECK-NOT: memref.copy
//

// ----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
func.func @dynamic_offset_strides(%x: memref<*xf32>, %y: memref<*xf32>, %z: memref<*xf32>,
                           %a : i32, %b : i32, %c : i32, %r : i32) {
  %ai = arith.index_cast %a : i32 to index
  %bi = arith.index_cast %b : i32 to index
  %ci = arith.index_cast %c : i32 to index
  %rand = arith.index_cast %r : i32 to index

  %offset  = arith.muli %ai, %rand : index
  %stride0  = arith.muli %bi, %rand : index
  %stride1  = arith.muli %ci, %rand : index

  %x_memref = memref.reinterpret_cast %x to offset: [%offset], sizes: [256, 64], strides: [%stride0, %stride1]
                         : memref<*xf32> to memref<256x64xf32, strided<[?, ?], offset: ?>>
  %y_memref = memref.reinterpret_cast %y to offset: [%offset], sizes: [256, 64], strides: [%stride0, %stride1]
                         : memref<*xf32> to memref<256x64xf32, strided<[?, ?], offset: ?>>
  %alloc = memref.alloc() : memref<256x64xf32>

  %x_tensor = bufferization.to_tensor %x_memref restrict : memref<256x64xf32, strided<[?, ?], offset: ?>> to tensor<256x64xf32>
  %y_tensor = bufferization.to_tensor %y_memref restrict : memref<256x64xf32, strided<[?, ?], offset: ?>> to tensor<256x64xf32>
  %alloc_t = bufferization.to_tensor %alloc restrict writable : memref<256x64xf32> to tensor<256x64xf32>

  %res = linalg.generic {indexing_maps = [#map1, #map1, #map1],
                            iterator_types = ["parallel", "parallel"]}
                  ins(%x_tensor, %y_tensor : tensor<256x64xf32>, tensor<256x64xf32>) outs(%alloc_t : tensor<256x64xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %24 = arith.divf %in, %in_5 : f32
    linalg.yield %24 : f32
  } -> tensor<256x64xf32>

  %z_memref = memref.reinterpret_cast %z to offset: [%offset], sizes: [256, 64], strides: [%stride0, %stride1]
                                : memref<*xf32> to memref<256x64xf32, strided<[?, ?], offset: ?>>
  bufferization.materialize_in_destination %res in writable %z_memref
                                : (tensor<256x64xf32>, memref<256x64xf32, strided<[?, ?], offset: ?>>) -> ()
  return
}

// CHECK-LABEL: @dynamic_offset_strides
// CHECK: %[[X:.+]]: memref<*xf32>, %[[Y:.+]]: memref<*xf32>, %[[Z:.+]]: memref<*xf32>, %[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32, %[[R:.+]]: i32
//
// CHECK: %[[Ai:.+]] = arith.index_cast %[[A]] : i32 to index
// CHECK: %[[Bi:.+]] = arith.index_cast %[[B]] : i32 to index
// CHECK: %[[Ci:.+]] = arith.index_cast %[[C]] : i32 to index
// CHECK: %[[Ri:.+]] = arith.index_cast %[[R]] : i32 to index
//
// CHECK: %[[OFFSET:.+]]  = arith.muli %[[Ai]], %[[Ri]] : index
// CHECK: %[[STRIDE0:.+]]  = arith.muli %[[Bi]], %[[Ri]] : index
// CHECK: %[[STRIDE1:.+]]  = arith.muli %[[Ci]], %[[Ri]] : index
//
// CHECK:  %[[CAST:.+]] = memref.reinterpret_cast %[[Z]] to offset: [%[[OFFSET]]], sizes: [256, 64], strides: [%[[STRIDE0]], %[[STRIDE1]]] : memref<*xf32> to memref<256x64xf32, strided<[?, ?], offset: ?>>
// CHECK:  linalg.generic {{.*}} ins({{.*}}) outs(%[[CAST]] : memref<256x64xf32, strided<[?, ?], offset: ?>>)
// CHECK-NOT: memref.copy
//

// ----

// Negative test: boundary-tile pattern where the reinterpret_cast has 2 uses
// (materialize in if-branch + subview in else-branch).  The pass must NOT add
// restrict because the cast result is aliased through the subview path.
func.func @boundary_tile_multi_use(%X: memref<*xf32>, %Y: memref<*xf32>,
                                   %Z: memref<*xf32>, %cond: i1,
                                   %dyn: index) {
  %reinterpret_cast_x = memref.reinterpret_cast %X to offset: [0], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1]>>
  %0 = bufferization.to_tensor %reinterpret_cast_x restrict
      : memref<128xf32, strided<[1]>> to tensor<128xf32>

  %reinterpret_cast_y = memref.reinterpret_cast %Y to offset: [0], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1]>>
  %1 = bufferization.to_tensor %reinterpret_cast_y restrict
      : memref<128xf32, strided<[1]>> to tensor<128xf32>

  %2 = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
                      iterator_types = ["parallel"]}
                      ins(%0, %1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %3 = arith.addf %in, %in_2 : f32
    linalg.yield %3 : f32
  } -> tensor<128xf32>

  // Cast is used by BOTH the materialize (if-branch) and subview (else-branch).
  %reinterpret_cast_z = memref.reinterpret_cast %Z to offset: [0], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1]>>
  scf.if %cond {
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_z
        : (tensor<128xf32>, memref<128xf32, strided<[1]>>) -> ()
  } else {
    %sub = memref.subview %reinterpret_cast_z[0] [%dyn] [1]
        : memref<128xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    %slice = tensor.extract_slice %2[0] [%dyn] [1] : tensor<128xf32> to tensor<?xf32>
    bufferization.materialize_in_destination %slice in writable %sub
        : (tensor<?xf32>, memref<?xf32, strided<[1]>>) -> ()
  }
  return
}

// restrict must NOT be added -> copy elision cannot fire -> memref.copy remains.
// CHECK-LABEL: @boundary_tile_multi_use
// CHECK: memref.copy

// ----

// NSP-sharded tile: each worker materializes its tile into a single-use
// subview of the output buffer.  The column tile is strided, so without copy
// elision the result would be copied element-wise by a strided memref.copy.
func.func @subview_tile(%X: memref<*xf32>, %Z: memref<*xf32>, %col: index) {
  %reinterpret_cast_x = memref.reinterpret_cast %X to offset: [0], sizes: [64, 32], strides: [32, 1]
      : memref<*xf32> to memref<64x32xf32, strided<[32, 1]>>
  %0 = bufferization.to_tensor %reinterpret_cast_x restrict
      : memref<64x32xf32, strided<[32, 1]>> to tensor<64x32xf32>
  %empty = tensor.empty() : tensor<64x32xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(i, j)->(i, j)>, affine_map<(i, j)->(i, j)>],
                      iterator_types = ["parallel", "parallel"]}
                      ins(%0 : tensor<64x32xf32>) outs(%empty : tensor<64x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<64x32xf32>
  %reinterpret_cast_z = memref.reinterpret_cast %Z to offset: [0], sizes: [64, 128], strides: [128, 1]
      : memref<*xf32> to memref<64x128xf32, strided<[128, 1]>>
  %sub = memref.subview %reinterpret_cast_z[0, %col] [64, 32] [1, 1]
      : memref<64x128xf32, strided<[128, 1]>> to memref<64x32xf32, strided<[128, 1], offset: ?>>
  bufferization.materialize_in_destination %1 in writable %sub
      : (tensor<64x32xf32>, memref<64x32xf32, strided<[128, 1], offset: ?>>) -> ()
  return
}

// CHECK-LABEL: @subview_tile
// CHECK: %[[SUB:.+]] = memref.subview
// CHECK: linalg.generic {{.*}} outs(%[[SUB]] : memref<64x32xf32, strided<[128, 1], offset: ?>>)
// CHECK-NOT: memref.copy
// CHECK: return
