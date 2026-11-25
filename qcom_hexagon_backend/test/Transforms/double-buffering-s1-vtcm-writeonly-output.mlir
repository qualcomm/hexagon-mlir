//===- double-buffering-s1-vtcm-writeonly-output.mlir ---------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt --pass-pipeline='builtin.module( \
// RUN:   func.func(vtcm-tiling{tile-size-override=131072 vtcm-budget=524288}), \
// RUN:   canonicalize,cse, \
// RUN:   hexagon-tiling,hexagon-vectorization,hexagon-vector-lowering, \
// RUN:   canonicalize,cse, \
// RUN:   one-shot-bufferize{bufferize-function-boundaries=1 allow-return-allocs-from-loops=1}, \
// RUN:   cse,canonicalize, \
// RUN:   func.func(hexagon-double-buffer-generic-s1),cse)' %s | FileCheck %s

// Test that double-buffering activates on a vtcm-tiled loop that has a
// write-only output (tensor.empty()).  vtcm-tiling produces a bare memref.alloc
// (no preceding subview/copy) for the output tile.  Double buffering must
// recognise this pattern and create ping-pong allocs for the output without
// attempting a no-op prefetch.

// CHECK-LABEL: func.func @add_kernel

// Six aligned ping-pong VTCM allocs: ping+pong for input-0, input-1, output.
// CHECK-COUNT-6: memref.alloc() {alignment = 2048 : i64} : memref<131072xf32, 1>

// Prologue: copy only the two read inputs into the ping buffers.
// The write-only output has no prefetch copy.
// CHECK:      scf.if %{{.*}} {
// CHECK-COUNT-2: memref.copy
// CHECK:      } {db_generic = 0 : i64, db_prologue}

// Main double-buffer loop.
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

// Ping kernel: prefetch copies only inputs (2), compute, then store output.
// CHECK:        scf.if %{{.*}} {
// CHECK:          scf.if %{{.*}} {
// CHECK-COUNT-2:    memref.copy
// CHECK:          } {db_prefetch}
// CHECK:          scf.for
// CHECK:            arith.addf
// CHECK:          memref.subview
// CHECK:          memref.copy %{{.*}}, %{{.*}} : memref<131072xf32, 1> to memref<131072xf32, strided<[1], offset: ?>>
// CHECK:        } {db_ping_kernel}

// Pong kernel: same structure.
// CHECK:        scf.if %{{.*}} {
// CHECK:          scf.if %{{.*}} {
// CHECK-COUNT-2:    memref.copy
// CHECK:          } {db_prefetch}
// CHECK:          scf.for
// CHECK:            arith.addf
// CHECK:          memref.subview
// CHECK:          memref.copy %{{.*}}, %{{.*}} : memref<131072xf32, 1> to memref<131072xf32, strided<[1], offset: ?>>
// CHECK:        } {db_pong_kernel}

// CHECK:      } {db_generic = 0 : i64}

#map = affine_map<(d0) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [524288], strides: [1] : memref<*xf32> to memref<524288xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [524288], strides: [1] : memref<*xf32> to memref<524288xf32, strided<[1]>>
    %0 = bufferization.to_tensor %reinterpret_cast_0 restrict : memref<524288xf32, strided<[1]>> to tensor<524288xf32>
    // tensor.empty() produces a write-only output: vtcm-tiling will generate a
    // bare memref.alloc (no subview/copy) for this tile.
    %1 = tensor.empty() : tensor<524288xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [524288], strides: [1] : memref<*xf32> to memref<524288xf32, strided<[1]>>
    %2 = bufferization.to_tensor %reinterpret_cast_1 restrict : memref<524288xf32, strided<[1]>> to tensor<524288xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %2 : tensor<524288xf32>, tensor<524288xf32>) outs(%1 : tensor<524288xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.addf %in, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<524288xf32>
    bufferization.materialize_in_destination %3 in restrict writable %reinterpret_cast : (tensor<524288xf32>, memref<524288xf32, strided<[1]>>) -> ()
    return
  }
}
