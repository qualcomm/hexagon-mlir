// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(seed-layout-conversions))' | FileCheck %s
// CHECK: func.func @main(%[[ARG0:.+]]: tensor<1x15x7x59xf16>, %[[ARG1:.+]]: tensor<59x1x1x59xf16>, %[[ARG2:.+]]: tensor<64x3x3x59xf16>, %[[ARG3:.+]]: tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16> {
// CHECK: %[[CST0:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] padding_value(%[[CST0]] : f16) inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into [[_:.*]] : tensor<1x15x7x59xf16> -> tensor<1x2x2x2x8x4x32xf16>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[PACK0]] inner_dims_pos = [5] inner_tiles = [2] into [[_:.*]] : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x2x2x2x8x2x32x2xf16>
// CHECK: %[[TO_ORIG_INPUT:.+]] = builtin.unrealized_conversion_cast %[[PACK1]] : tensor<1x2x2x2x8x2x32x2xf16> to tensor<1x15x7x59xf16>
// CHECK: %[[PADDED_FILTER:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[5, 0, 0, 5] {
// CHECK: ^bb0([[_:.*]]: index, [[_:.*]]: index, [[_:.*]]: index, [[_:.*]]: index):
// CHECK: tensor.yield %[[CST0]] : f16
// CHECK: } : tensor<59x1x1x59xf16> to tensor<64x1x1x64xf16>
// CHECK: %[[RESHAPED:.+]] = tensor.reshape %[[PADDED_FILTER]]([[_:.*]]) : (tensor<64x1x1x64xf16>, tensor<7xindex>) -> tensor<2x32x1x1x2x16x2xf16>
// CHECK: %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[RESHAPED]] : tensor<2x32x1x1x2x16x2xf16>) outs([[_:.*]] : tensor<2x2x1x1x16x32x2xf16>) permutation = [0, 4, 2, 3, 5, 1, 6]
// CHECK: %[[TO_ORIG_FILTER:.+]] = builtin.unrealized_conversion_cast %[[TRANSPOSED]] : tensor<2x2x1x1x16x32x2xf16> to tensor<59x1x1x59xf16>
// CHECK: %[[CONV_OUTPUT:.+]] = linalg.conv_2d_nhwc_fhwc ins(%[[TO_ORIG_INPUT]], %[[TO_ORIG_FILTER]] : tensor<1x15x7x59xf16>, tensor<59x1x1x59xf16>) outs([[_:.*]] : tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
// CHECK: %[[TO_PACKED_OUTPUT:.+]] = builtin.unrealized_conversion_cast %[[CONV_OUTPUT]] : tensor<1x15x7x59xf16> to tensor<1x2x2x2x8x2x32x2xf16>
// CHECK: %[[UNPACK0:.+]] = linalg.unpack %[[TO_PACKED_OUTPUT]] inner_dims_pos = [5] inner_tiles = [2] into [[_:.*]] : tensor<1x2x2x2x8x2x32x2xf16> -> tensor<1x2x2x2x8x4x32xf16>
// CHECK: %[[UNPACK1:.+]] = linalg.unpack %[[UNPACK0]] inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into [[_:.*]] : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x15x7x59xf16>
// CHECK: %[[RET_VAL:.+]] = linalg.conv_2d_nhwc_fhwc {strides = dense<2> : vector<2xi64>} ins(%[[UNPACK1]], %[[ARG2]] : tensor<1x15x7x59xf16>, tensor<64x3x3x59xf16>) outs(%[[ARG3]] : tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16>
// CHECK: return %[[RET_VAL]] : tensor<1x7x3x64xf16>
module {
func.func @main(%in1: tensor<1x15x7x59xf16>,
               %in2: tensor<59x1x1x59xf16>,
               %in3: tensor<64x3x3x59xf16>,
               %result: tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16> {
  %empty = tensor.empty() : tensor<1x15x7x59xf16>
  %0 = linalg.conv_2d_nhwc_fhwc ins(%in1, %in2 : tensor<1x15x7x59xf16>, tensor<59x1x1x59xf16>) outs(%empty:  tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
  %1 = linalg.conv_2d_nhwc_fhwc
    { strides = dense<[2, 2]> : vector<2xi64> }
    ins(%0, %in3 : tensor<1x15x7x59xf16>, tensor<64x3x3x59xf16>) outs(%result: tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16>
  return %1 : tensor<1x7x3x64xf16>
}

}
