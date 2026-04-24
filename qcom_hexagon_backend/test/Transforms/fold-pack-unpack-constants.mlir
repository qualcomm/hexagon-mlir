// RUN: linalg-hexagon-opt %s -split-input-file --fold-pack-unpack-constants | FileCheck %s

// -----
// Test: Fold linalg.pack on constant tensor (1D -> 2D)
// CHECK-LABEL: func.func @fold_pack_1d_to_2d
func.func @fold_pack_1d_to_2d() -> tensor<3x4xf32> {
  // CHECK-NOT: linalg.pack
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]> : tensor<3x4xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]> : tensor<12xf32>
  %empty = tensor.empty() : tensor<3x4xf32>
  %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [4] into %empty : tensor<12xf32> -> tensor<3x4xf32>
  return %packed : tensor<3x4xf32>
}

// -----
// Test: Fold linalg.pack on constant tensor (2D -> 4D)
// CHECK-LABEL: func.func @fold_pack_2d_to_4d
func.func @fold_pack_2d_to_4d() -> tensor<2x2x2x2xf32> {
  // CHECK-NOT: linalg.pack
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}{{\[}}{{\[}}[0.000000e+00, 1.000000e+00], [4.000000e+00, 5.000000e+00]], {{\[}}[2.000000e+00, 3.000000e+00], [6.000000e+00, 7.000000e+00]]], {{\[}}{{\[}}[8.000000e+00, 9.000000e+00], [1.200000e+01, 1.300000e+01]], {{\[}}[1.000000e+01, 1.100000e+01], [1.400000e+01, 1.500000e+01]]]]> : tensor<2x2x2x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]> : tensor<4x4xf32>
  %empty = tensor.empty() : tensor<2x2x2x2xf32>
  %packed = linalg.pack %cst inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %empty : tensor<4x4xf32> -> tensor<2x2x2x2xf32>
  return %packed : tensor<2x2x2x2xf32>
}

// -----
// Test: Fold linalg.pack with padding
// CHECK-LABEL: func.func @fold_pack_with_padding
func.func @fold_pack_with_padding() -> tensor<2x4xf16> {
  // CHECK-NOT: linalg.pack
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<2x4xf16>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>
  %pad = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<2x4xf16>
  %packed = linalg.pack %cst padding_value(%pad : f16) inner_dims_pos = [0] inner_tiles = [4] into %empty : tensor<5xf16> -> tensor<2x4xf16>
  return %packed : tensor<2x4xf16>
}

// -----
// Test: Fold linalg.unpack on constant tensor (2D -> 1D)
// CHECK-LABEL: func.func @fold_unpack_2d_to_1d
func.func @fold_unpack_2d_to_1d() -> tensor<12xf32> {
  // CHECK-NOT: linalg.unpack
  // CHECK: %[[CST:.+]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]> : tensor<12xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]> : tensor<3x4xf32>
  %empty = tensor.empty() : tensor<12xf32>
  %unpacked = linalg.unpack %cst inner_dims_pos = [0] inner_tiles = [4] into %empty : tensor<3x4xf32> -> tensor<12xf32>
  return %unpacked : tensor<12xf32>
}

// -----
// Test: Fold linalg.unpack on constant tensor (4D -> 2D)
// CHECK-LABEL: func.func @fold_unpack_4d_to_2d
func.func @fold_unpack_4d_to_2d() -> tensor<4x4xf32> {
  // CHECK-NOT: linalg.unpack
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 4.000000e+00, 5.000000e+00], [2.000000e+00, 3.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.200000e+01, 1.300000e+01], [1.000000e+01, 1.100000e+01, 1.400000e+01, 1.500000e+01]]> : tensor<4x4xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], [[[8.0, 9.0], [10.0, 11.0]], [[12.0, 13.0], [14.0, 15.0]]]]> : tensor<2x2x2x2xf32>
  %empty = tensor.empty() : tensor<4x4xf32>
  %unpacked = linalg.unpack %cst inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %empty : tensor<2x2x2x2xf32> -> tensor<4x4xf32>
  return %unpacked : tensor<4x4xf32>
}

// -----
// Test: Fold linalg.transpose on constant tensor (2D)
// CHECK-LABEL: func.func @fold_transpose_2d
func.func @fold_transpose_2d() -> tensor<3x2xf32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %empty = tensor.empty() : tensor<3x2xf32>
  %transposed = linalg.transpose ins(%cst : tensor<2x3xf32>) outs(%empty : tensor<3x2xf32>) permutation = [1, 0]
  return %transposed : tensor<3x2xf32>
}

// -----
// Test: Fold linalg.transpose on constant tensor (3D)
// CHECK-LABEL: func.func @fold_transpose_3d
func.func @fold_transpose_3d() -> tensor<3x2x2xf32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<3x2x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]]> : tensor<2x2x3xf32>
  %empty = tensor.empty() : tensor<3x2x2xf32>
  %transposed = linalg.transpose ins(%cst : tensor<2x2x3xf32>) outs(%empty : tensor<3x2x2xf32>) permutation = [2, 0, 1]
  return %transposed : tensor<3x2x2xf32>
}

// -----
// Test: Fold tensor.expand_shape on constant tensor
// CHECK-LABEL: func.func @fold_expand_shape
func.func @fold_expand_shape() -> tensor<2x3x2xf32> {
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x3x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]> : tensor<2x6xf32>
  %expanded = tensor.expand_shape %cst [[0], [1, 2]] output_shape [2, 3, 2] : tensor<2x6xf32> into tensor<2x3x2xf32>
  return %expanded : tensor<2x3x2xf32>
}

// -----
// Test: Fold tensor.collapse_shape on constant tensor
// CHECK-LABEL: func.func @fold_collapse_shape
func.func @fold_collapse_shape() -> tensor<2x6xf32> {
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]> : tensor<2x6xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]]> : tensor<2x3x2xf32>
  %collapsed = tensor.collapse_shape %cst [[0], [1, 2]] : tensor<2x3x2xf32> into tensor<2x6xf32>
  return %collapsed : tensor<2x6xf32>
}

// -----
// Test: Fold tensor.pad on constant tensor
// CHECK-LABEL: func.func @fold_pad
func.func @fold_pad() -> tensor<5x5xf32> {
  // CHECK-NOT: tensor.pad
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 0.000000e+00], [0.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 0.000000e+00], [0.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<5x5xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]> : tensor<3x3xf32>
  %pad_value = arith.constant 0.0 : f32
  %padded = tensor.pad %cst low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_value : f32
  } : tensor<3x3xf32> to tensor<5x5xf32>
  return %padded : tensor<5x5xf32>
}

// -----
// Test: Fold tensor.pad with asymmetric padding
// CHECK-LABEL: func.func @fold_pad_asymmetric
func.func @fold_pad_asymmetric() -> tensor<4x6xf16> {
  // CHECK-NOT: tensor.pad
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00], [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]]> : tensor<4x6xf16>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>
  %pad_value = arith.constant -1.0 : f16
  %padded = tensor.pad %cst low[1, 0] high[1, 3] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_value : f16
  } : tensor<2x3xf16> to tensor<4x6xf16>
  return %padded : tensor<4x6xf16>
}

// -----
// Test: Fold pack followed by unpack (round-trip)
// CHECK-LABEL: func.func @fold_pack_unpack_roundtrip
func.func @fold_pack_unpack_roundtrip() -> tensor<8xf32> {
  // CHECK-NOT: linalg.pack
  // CHECK-NOT: linalg.unpack
  // CHECK: %[[CST:.+]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>
  %empty1 = tensor.empty() : tensor<2x4xf32>
  %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [4] into %empty1 : tensor<8xf32> -> tensor<2x4xf32>
  %empty2 = tensor.empty() : tensor<8xf32>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0] inner_tiles = [4] into %empty2 : tensor<2x4xf32> -> tensor<8xf32>
  return %unpacked : tensor<8xf32>
}

// -----
// Test: Fold multiple operations in sequence
// CHECK-LABEL: func.func @fold_multiple_ops
func.func @fold_multiple_ops() -> tensor<2x3xf32> {
  // CHECK-NOT: tensor.expand_shape
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0.000000e+00, 2.000000e+00, 4.000000e+00], [1.000000e+00, 3.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<6xf32>
  %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  %empty = tensor.empty() : tensor<2x3xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<3x2xf32>) outs(%empty : tensor<2x3xf32>) permutation = [1, 0]
  return %transposed : tensor<2x3xf32>
}

// -----
// Test: Integer types (i8)
// CHECK-LABEL: func.func @fold_pack_i8
func.func @fold_pack_i8() -> tensor<2x4xi8> {
  // CHECK-NOT: linalg.pack
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi8>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
  %empty = tensor.empty() : tensor<2x4xi8>
  %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [4] into %empty : tensor<8xi8> -> tensor<2x4xi8>
  return %packed : tensor<2x4xi8>
}

// -----
// Test: Integer types (i32)
// CHECK-LABEL: func.func @fold_transpose_i32
func.func @fold_transpose_i32() -> tensor<3x2xi32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[CST:.+]] = arith.constant dense<{{\[}}[0, 3], [1, 4], [2, 5]]> : tensor<3x2xi32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %empty = tensor.empty() : tensor<3x2xi32>
  %transposed = linalg.transpose ins(%cst : tensor<2x3xi32>) outs(%empty : tensor<3x2xi32>) permutation = [1, 0]
  return %transposed : tensor<3x2xi32>
}

// -----
// Test: f16 type
// CHECK-LABEL: func.func @fold_expand_shape_f16
func.func @fold_expand_shape_f16() -> tensor<2x2x2xf16> {
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x2x2xf16>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf16>
  %expanded = tensor.expand_shape %cst [[0], [1, 2]] output_shape [2, 2, 2] : tensor<2x4xf16> into tensor<2x2x2xf16>
  return %expanded : tensor<2x2x2xf16>
}

// -----
// Test: Fold pack on splat constant
// CHECK-LABEL: func.func @fold_pack_splat
func.func @fold_pack_splat() -> tensor<2x4xf32> {
  // CHECK-NOT: linalg.pack
  // CHECK: %[[CST:.+]] = arith.constant dense<5.000000e+00> : tensor<2x4xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<5.0> : tensor<8xf32>
  %empty = tensor.empty() : tensor<2x4xf32>
  %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [4] into %empty : tensor<8xf32> -> tensor<2x4xf32>
  return %packed : tensor<2x4xf32>
}

// -----
// Test: Fold transpose on splat constant
// CHECK-LABEL: func.func @fold_transpose_splat
func.func @fold_transpose_splat() -> tensor<3x2xf32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : tensor<3x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<1.0> : tensor<2x3xf32>
  %empty = tensor.empty() : tensor<3x2xf32>
  %transposed = linalg.transpose ins(%cst : tensor<2x3xf32>) outs(%empty : tensor<3x2xf32>) permutation = [1, 0]
  return %transposed : tensor<3x2xf32>
}

// -----
// Test: Fold pad on splat constant
// CHECK-LABEL: func.func @fold_pad_splat
func.func @fold_pad_splat() -> tensor<5x5xf32> {
  // CHECK-NOT: tensor.pad
  // CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<5x5xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<2.0> : tensor<3x3xf32>
  %pad_value = arith.constant 0.0 : f32
  %padded = tensor.pad %cst low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_value : f32
  } : tensor<3x3xf32> to tensor<5x5xf32>
  return %padded : tensor<5x5xf32>
}

