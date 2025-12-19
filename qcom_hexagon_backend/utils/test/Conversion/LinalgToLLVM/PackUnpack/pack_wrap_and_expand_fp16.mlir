// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(seed-layout-conversions,extend-pack-frontier{parallels-only=false}))' | FileCheck %s
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%in1: tensor<1x15x7x59x10xf16>,
              %in2: tensor<59x1x1x59xf16>,
              %in3: tensor<64x3x3x59xf16>,
              %result: tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16> {
  %reduce_dst = tensor.empty() : tensor<1x15x7x59xf16>
  %reduce = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} 
      ins(%in1 : tensor<1x15x7x59x10xf16>)
      outs(%reduce_dst : tensor<1x15x7x59xf16>) {
  ^bb0(%in_one: f16, %out_one: f16) :
      %red = arith.addf %in_one, %out_one: f16
      linalg.yield %red : f16
  } -> tensor<1x15x7x59xf16>
  %empty = tensor.empty() : tensor<1x15x7x59xf16>
  %empty3 = tensor.empty() : tensor<1x15x7x59xf16>
  %addfive = linalg.generic {
      indexing_maps = [#map2, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
      ins(%reduce : tensor<1x15x7x59xf16>)
      outs(%empty3 : tensor<1x15x7x59xf16>) {
  ^bb0(%in_one: f16, %out_one: f16) :
      %c5 = arith.constant 5.0 : f16
      %res = arith.addf %c5, %in_one: f16
      linalg.yield %res : f16
  } -> tensor<1x15x7x59xf16>
  %0 = linalg.conv_2d_nhwc_fhwc ins(%addfive, %in2 : tensor<1x15x7x59xf16>, tensor<59x1x1x59xf16>)
      outs(%empty:  tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>

  %empty4 = tensor.empty() : tensor<1x15x7x59xf16>
  %post_first_conv_elementwise = linalg.generic {
      indexing_maps = [#map2, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
      ins(%0 : tensor<1x15x7x59xf16>)
      outs(%empty4 : tensor<1x15x7x59xf16>) {
  ^bb0(%in_one: f16, %out_one: f16) :
      %c5 = arith.constant 5.0 : f16
      %res = arith.addf %c5, %in_one: f16
      linalg.yield %res : f16
  } -> tensor<1x15x7x59xf16>
  
  %1 = linalg.conv_2d_nhwc_fhwc
      {strides = dense<[2, 2]> : vector<2xi64>}
      ins(%post_first_conv_elementwise, %in3 : tensor<1x15x7x59xf16>, tensor<64x3x3x59xf16>)
      outs(%result: tensor<1x7x3x64xf16>) -> tensor<1x7x3x64xf16>
  return %1 : tensor<1x7x3x64xf16>
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME: %[[A0:.+]]: tensor<1x15x7x59x10xf16>, %[[A1:.+]]: tensor<59x1x1x59xf16>, %[[A2:.+]]: tensor<64x3x3x59xf16>, %[[A3:.+]]: tensor<1x7x3x64xf16>)
// CHECK: %[[PACK_A:.*]] = linalg.pack %[[A0]]
// CHECK: %[[PACK_B:.*]] = linalg.pack %[[PACK_A]]
// CHECK: %[[REDUCTION_PRE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[PACK_B]] : tensor<1x2x2x2x10x8x2x32x2xf16>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[REDUCTION_PRE]] : tensor<1x2x2x2x8x2x32x2xf16>)
// CHECK: %[[GENERIC_POST:.*]] = linalg.generic
// CHECK: %[[UNPACK_A:.*]] = linalg.unpack %[[GENERIC_POST]]
// CHECK: %[[UNPACK_B:.*]] = linalg.unpack %[[UNPACK_A]]
// CHECK: %[[CONV_FINAL:.*]] = linalg.conv_2d_nhwc_fhwc
// CHECK-SAME: ins(%[[UNPACK_B]], %[[A2]]
// CHECK: return %[[CONV_FINAL]]
