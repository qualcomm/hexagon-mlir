//===- input.mlir ---------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#map5  = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>

func.func @vectorize_sum_of_squares_mlir(
    %arg0: memref<*xf16> {tt.divisibility = 16 : i32},
    %arg1: memref<*xf16> {tt.divisibility = 16 : i32},
    %arg2: memref<*xf16> {tt.divisibility = 16 : i32},
    %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32,
    %arg7: i32, %arg8: i32, %arg9: i32) {
  %c0 = arith.constant 0 : index

  %in0 = memref.reinterpret_cast %arg0 to offset: [%c0],
      sizes: [2, 24, 8, 2, 32, 2],
      strides: [24576, 1024, 128, 64, 2, 1]
      : memref<*xf16> to memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>

  %in1 = memref.reinterpret_cast %arg1 to offset: [%c0],
      sizes: [2, 24, 8, 2, 32, 2],
      strides: [24576, 1024, 128, 64, 2, 1]
      : memref<*xf16> to memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>

  %out_memref = memref.reinterpret_cast %arg2 to offset: [%c0],
      sizes: [2, 8, 2, 2],
      strides: [32, 4, 2, 1]
      : memref<*xf16> to memref<2x8x2x2xf16, strided<[32, 4, 2, 1], offset: ?>>

  %in0_t = bufferization.to_tensor %in0 restrict
    : memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>
    to tensor<2x24x8x2x32x2xf16>
  %in1_t = bufferization.to_tensor %in1 restrict
    : memref<2x24x8x2x32x2xf16, strided<[24576, 1024, 128, 64, 2, 1], offset: ?>>
    to tensor<2x24x8x2x32x2xf16>

  %init = tensor.empty() : tensor<2x8x2x2xf16>
  %zero = arith.constant 0.0 : f16
  %filled = linalg.fill ins(%zero : f16) outs(%init : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>

  %res = linalg.generic {
      indexing_maps = [#map5, #map5, #map10],
      iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
    } ins(%in0_t, %in1_t : tensor<2x24x8x2x32x2xf16>, tensor<2x24x8x2x32x2xf16>)
      outs(%filled : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %in_136: f16, %out: f16):
      %0 = arith.mulf %in, %in_136 : f16
      %1 = arith.addf %0, %out : f16
      linalg.yield %1 : f16
    } -> tensor<2x8x2x2xf16>

  bufferization.materialize_in_destination %res in writable %out_memref
    : (tensor<2x8x2x2xf16>, memref<2x8x2x2xf16, strided<[32, 4, 2, 1], offset: ?>>) -> ()
  return
}
