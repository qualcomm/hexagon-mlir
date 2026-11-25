//===- empty_out_no_ddr_copy.mlir -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Verify that a tensor.empty() output does not produce a DDR alloc + copy.
// The output block arg is read in the body (arith.addf), so isOutsWriteOnly
// is false and copyToVTCM is called — the fix suppresses the DDR copy.
//
// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:     -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s

// -----

#map  = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
func.func @reduction_empty_out(%x: memref<128x512xf32>, %z: memref<128xf32>) {
  %t0    = bufferization.to_tensor %x restrict writable : memref<128x512xf32> to tensor<128x512xf32>
  %empty = tensor.empty() : tensor<128xf32>
  // %out is read as the reduction accumulator — isOutsWriteOnly returns false,
  // so copyToVTCM is called with tensor.empty() as source.
  // The fix in copyToMemorySpace suppresses the copy operand.
  %r0 = linalg.generic {
          indexing_maps = [#map, #map1],
          iterator_types = ["parallel", "reduction"]}
          ins(%t0 : tensor<128x512xf32>)
          outs(%empty : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<128xf32>
  bufferization.materialize_in_destination %r0 in writable %z
      : (tensor<128xf32>, memref<128xf32>) -> ()
  return
}
}

// CHECK-LABEL: @reduction_empty_out
// Input is copied from DDR to VTCM.
// CHECK: %[[IN:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x512xf32, 1>
// CHECK-NEXT: memref.copy %{{.+}}, %[[IN]] : memref<128x512xf32> to memref<128x512xf32, 1>
// Output is allocated on VTCM with no copy from DDR — the fix suppresses the
// copy operand that copyToVTCM would otherwise emit for tensor.empty().
// CHECK: %[[OUT:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
// CHECK-NOT: memref.copy %{{.+}}, %[[OUT]]
// Compute runs on VTCM buffers.
// CHECK: linalg.generic {{.*}} ins(%[[IN]] : memref<128x512xf32, 1>) outs(%[[OUT]] : memref<128xf32, 1>)
