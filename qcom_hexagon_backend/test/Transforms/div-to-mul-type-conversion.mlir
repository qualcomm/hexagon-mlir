// RUN: linalg-hexagon-opt %s -div-to-mul-optimization -cse -canonicalize | FileCheck %s

// This test verifies div-to-mul optimization when division operations are wrapped in
// type conversion operations (extf/truncf). The optimization computes the reciprocal
// at the original precision (f16), then performs multiplication at the extended precision
// (f32), preserving the type conversion pattern while eliminating the division.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @div_to_mul_with_type_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<128xf16>, %[[ARG1:.*]]: tensor<f16>) -> tensor<128xf16>
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128xf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[ARG1]][] : tensor<f16>
// CHECK: %[[RECIP:.*]] = arith.divf %[[CST]], %[[EXTRACTED]] : f16
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[RECIP]] : f16) outs(%[[EMPTY]] : tensor<128xf16>) -> tensor<128xf16>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%[[ARG0]], %[[FILLED]] : tensor<128xf16>, tensor<128xf16>)
// CHECK-SAME: outs(%[[ARG0]] : tensor<128xf16>)
// CHECK: ^bb0(%[[IN:.*]]: f16, %[[IN_0:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:   %[[EXT1:.*]] = arith.extf %[[IN]] : f16 to f32
// CHECK:   %[[EXT2:.*]] = arith.extf %[[IN_0]] : f16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[EXT1]], %[[EXT2]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %[[MUL]] : f32 to f16
// CHECK:   linalg.yield %[[TRUNC]] : f16
// CHECK: return %[[RESULT]]

func.func @div_to_mul_with_type_conversion(%arg0: tensor<128xf16>, %arg1: tensor<f16>) -> tensor<128xf16> {
  %empty = tensor.empty() : tensor<128xf16>

  // Extract scalar from tensor
  %extracted = tensor.extract %arg1[] : tensor<f16>

  // Fill operation with extracted scalar
  %filled = linalg.fill ins(%extracted : f16) outs(%empty : tensor<128xf16>) -> tensor<128xf16>

  // Generic operation with division wrapped in type conversions
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %filled : tensor<128xf16>, tensor<128xf16>)
    outs(%arg0 : tensor<128xf16>) {
  ^bb0(%in: f16, %in_4: f16, %out: f16):
    %ext1 = arith.extf %in : f16 to f32
    %ext2 = arith.extf %in_4 : f16 to f32
    %div = arith.divf %ext1, %ext2 : f32
    %trunc = arith.truncf %div : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<128xf16>

  return %result : tensor<128xf16>
}
