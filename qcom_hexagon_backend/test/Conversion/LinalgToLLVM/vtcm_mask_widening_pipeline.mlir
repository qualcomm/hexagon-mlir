//===- vtcm_mask_widening_pipeline.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:     -pass-pipeline='builtin.module(func.func(vtcm-mask-widening,vtcm-tiling),canonicalize, \
// RUN:         one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s
//
// End-to-end check that VTCMMaskWideningPass and VTCMTilingPass, run
// back-to-back in pipeline order, select the same operands consistently for
// an i1 mask -- for both the small and large case.
//
// Both the widening cast (extui) and the consuming select are pure
// elementwise generics (identity indexing maps, all-parallel iterators), so
// VTCMTilingPass now skips staging either of them into VTCM regardless of
// size (see [VTCMTiling] Skip VTCM staging for pure elementwise generics).
// The only allocation left is the plain DDR-resident i8 buffer needed to
// hold the widened mask between the two generics.

// -----

// Small mask (prefetch path): the VTCM-resident buffer for the mask is i8,
// not i1, and the consuming generic truncs it back to i1 in its body.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_mask_prefetched(%mask: memref<128x256xi1>, %x: memref<128x256xf32>, %z: memref<128x256xf32>) {
   %tmask = bufferization.to_tensor %mask restrict writable : memref<128x256xi1> to tensor<128x256xi1>
   %tx = bufferization.to_tensor %x restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %tz = bufferization.to_tensor %z restrict writable : memref<128x256xf32> to tensor<128x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%tmask, %tx : tensor<128x256xi1>, tensor<128x256xf32>)
           outs(%tz : tensor<128x256xf32>) {
   ^bb0(%m: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %m, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<128x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<128x256xf32>, memref<128x256xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @i1_mask_prefetched
// Both generics are pure elementwise, so neither stages into VTCM -- the
// widened mask lives in a single plain DDR buffer shared by both ops.
// CHECK: %[[DDR_I8:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xi8>
// CHECK-NOT: memref.alloc(){{.*}}, 1>
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[DDR_I8]] : memref<128x256xi8>)
// CHECK: arith.extui
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[DDR_I8]]
// CHECK: ^bb0(%[[ARG:.+]]: i8,
// CHECK: %[[TRUNC:.+]] = arith.trunci %[[ARG]] : i8 to i1
// CHECK: arith.select %[[TRUNC]]
// CHECK: memref.dealloc %[[DDR_I8]] : memref<128x256xi8>

// -----

// Large mask (tiled/sliced path): the mask is widened to i8 before tiling,
// so each per-tile VTCM subview/copy is i8-typed too.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_mask_tiled(%mask: memref<4096x8192xi1>, %x: memref<4096x8192xf32>, %z: memref<4096x8192xf32>) {
   %tmask = bufferization.to_tensor %mask restrict writable : memref<4096x8192xi1> to tensor<4096x8192xi1>
   %tx = bufferization.to_tensor %x restrict writable : memref<4096x8192xf32> to tensor<4096x8192xf32>
   %tz = bufferization.to_tensor %z restrict writable : memref<4096x8192xf32> to tensor<4096x8192xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%tmask, %tx : tensor<4096x8192xi1>, tensor<4096x8192xf32>)
           outs(%tz : tensor<4096x8192xf32>) {
   ^bb0(%m: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %m, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<4096x8192xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<4096x8192xf32>, memref<4096x8192xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @i1_mask_tiled
// Both generics stay pure elementwise at this size too, so neither is tiled
// or staged into VTCM -- same plain single-buffer, single-generic shape as
// the prefetched case, just at a larger size.
// CHECK: %[[MASK_I8_DDR:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4096x8192xi8>
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[MASK_I8_DDR]] : memref<4096x8192xi8>)
// CHECK: arith.extui
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[MASK_I8_DDR]]
// CHECK: ^bb0(%[[ARG:.+]]: i8,
// CHECK: %[[TRUNC:.+]] = arith.trunci %[[ARG]] : i8 to i1
// CHECK: arith.select %[[TRUNC]]
// CHECK: memref.dealloc %[[MASK_I8_DDR]] : memref<4096x8192xi8>
