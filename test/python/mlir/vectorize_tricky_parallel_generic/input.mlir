//===- input.mlir ---------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>

func.func @vectorize_tricky_parallel_generic_mlir(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf16> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) {
  %c0 = arith.constant 0 : index
  %cst29 = arith.constant 1.000000e-05 : f64
  %cst31 = arith.constant 2.560000e+02 : f16

  %in = memref.reinterpret_cast %arg0 to offset: [%c0],
      sizes: [2, 24, 8, 2, 32, 2],
      strides: [24576, 1024, 128, 64, 2, 1]
      : memref<*xf16> to memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>

  %in136 = memref.reinterpret_cast %arg1 to offset: [%c0],
      sizes: [2, 8, 2, 2],
      strides: [32, 4, 2, 1]
      : memref<*xf16> to memref<2x8x2x2xf16, strided<[32, 4, 2, 1], offset: ?>>

  %in137 = memref.reinterpret_cast %arg2 to offset: [%c0],
      sizes: [24, 32],
      strides: [32, 1]
      : memref<*xf16> to memref<24x32xf16, strided<[32, 1], offset: ?>>

  %in138 = memref.reinterpret_cast %arg3 to offset: [%c0],
      sizes: [24, 32],
      strides: [32, 1]
      : memref<*xf16> to memref<24x32xf16, strided<[32, 1], offset: ?>>

  %out_memref = memref.reinterpret_cast %arg4 to offset: [%c0],
      sizes: [2, 24, 8, 2, 32, 2],
      strides: [24576, 1024, 128, 64, 2, 1]
      : memref<*xf16> to memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>

  %in_t = bufferization.to_tensor %in restrict
    : memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>
    to tensor<2x24x8x2x32x2xf16>
  %in136_t = bufferization.to_tensor %in136 restrict
    : memref<2x8x2x2xf16, strided<[32, 4, 2, 1], offset: ?>>
    to tensor<2x8x2x2xf16>
  %in137_t = bufferization.to_tensor %in137 restrict
    : memref<24x32xf16, strided<[32, 1], offset: ?>>
    to tensor<24x32xf16>
  %in138_t = bufferization.to_tensor %in138 restrict
    : memref<24x32xf16, strided<[32, 1], offset: ?>>
    to tensor<24x32xf16>

  %init = tensor.empty() : tensor<2x24x8x2x32x2xf16>
  %res = linalg.generic {
      indexing_maps = [#map5, #map10, #map6, #map6, #map5],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%in_t, %in136_t, %in137_t, %in138_t : tensor<2x24x8x2x32x2xf16>, tensor<2x8x2x2xf16>, tensor<24x32xf16>, tensor<24x32xf16>) outs(%init : tensor<2x24x8x2x32x2xf16>) {
    ^bb0(%in_elem: f16, %in136_elem: f16, %in137_elem: f16, %in138_elem: f16, %out_elem: f16):
      %0 = arith.divf %in136_elem, %cst31 : f16
      %1 = arith.truncf %cst29 : f64 to f16
      %2 = arith.addf %0, %1 : f16
      %3 = math.rsqrt %2 : f16
      %4 = arith.mulf %in_elem, %3 : f16
      %5 = arith.mulf %4, %in137_elem : f16
      %6 = arith.addf %5, %in138_elem : f16
      linalg.yield %6 : f16
    } -> tensor<2x24x8x2x32x2xf16>

  bufferization.materialize_in_destination %res in writable %out_memref
    : (tensor<2x24x8x2x32x2xf16>, memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>) -> ()
  return
}
