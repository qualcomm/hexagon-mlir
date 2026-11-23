//===- vtcm_tiling_write_only_out.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:     -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s
//
// Tests verifying that a pure elementwise add -- whose outs operand is
// write-only (never read in the body) -- skips VTCM staging entirely
// regardless of size, operating directly on the DDR-resident args instead.

// -----

// Small tensors that fit entirely in VTCM (prefetch path).
// C = A + B: %out is never read in the body, so no DDR->VTCM copy for C.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @write_only_out_small(%x: memref<128x256xf32>, %y: memref<128x256xf32>, %z: memref<128x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<128x256xf32> to tensor<128x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<128x256xf32>, tensor<128x256xf32>)
           outs(%t2 : tensor<128x256xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):
     %4 = arith.addf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<128x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<128x256xf32>, memref<128x256xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @write_only_out_small
// CHECK-SAME: %[[X:.+]]: memref<128x256xf32>, %[[Y:.+]]: memref<128x256xf32>, %[[Z:.+]]: memref<128x256xf32>
//
// This add is a pure elementwise generic, which VTCMTilingPass now skips
// staging into VTCM by default -- it operates directly on the DDR-resident
// args, in place.
// CHECK-NOT: memref.alloc
// CHECK: linalg.generic {{.*}} ins(%[[X]], %[[Y]] : memref<128x256xf32>, memref<128x256xf32>) outs(%[[Z]] : memref<128x256xf32>)
// CHECK-NOT: memref.dealloc

// -----

// Large tensors that exceed VTCM budget (tiling path).
// C = A + B: %out is never read in the body, so no DDR->VTCM copy for C tiles.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @write_only_out_large(%x: memref<4096x8192xf32>, %y: memref<4096x8192xf32>, %z: memref<4096x8192xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<4096x8192xf32> to tensor<4096x8192xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<4096x8192xf32> to tensor<4096x8192xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<4096x8192xf32> to tensor<4096x8192xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<4096x8192xf32>, tensor<4096x8192xf32>)
           outs(%t2 : tensor<4096x8192xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):
     %4 = arith.addf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<4096x8192xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<4096x8192xf32>, memref<4096x8192xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @write_only_out_large
// CHECK-SAME: %[[X:.+]]: memref<4096x8192xf32>, %[[Y:.+]]: memref<4096x8192xf32>, %[[Z:.+]]: memref<4096x8192xf32>
//
// This add is a pure elementwise generic, so VTCMTilingPass skips staging
// and tiling it regardless of size -- it operates directly on the
// DDR-resident args, in place, with no loop at all.
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.for
// CHECK: linalg.generic {{.*}} ins(%[[X]], %[[Y]] : memref<4096x8192xf32>, memref<4096x8192xf32>) outs(%[[Z]] : memref<4096x8192xf32>)
// CHECK-NOT: memref.dealloc
