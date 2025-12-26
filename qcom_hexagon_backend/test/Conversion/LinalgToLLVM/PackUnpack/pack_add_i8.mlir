// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-pack))' | FileCheck %s
// CHECK: func.func @main(%[[ARG0:.+]]: memref<1x20x16x14xi8>, %[[ARG1:.+]]: memref<1x20x16x14xi8>, %[[ARG2:.+]]: memref<1x20x16x14xi8>) {
// CHECK: %[[T0:.+]] = bufferization.to_tensor %[[ARG0]] restrict : memref<1x20x16x14xi8> to tensor<1x20x16x14xi8>
// CHECK: %[[T1:.+]] = bufferization.to_tensor %[[ARG1]] restrict : memref<1x20x16x14xi8> to tensor<1x20x16x14xi8>
// CHECK: %[[PADDED0:.+]] = tensor.pad %[[T0]] low[0, 0, 0, 0] high[0, 4, 0, 18]
// CHECK: %[[EXPANDED0:.+]] = tensor.expand_shape %[[PADDED0]] {{\[\[}}0], [1, 2], [3, 4], [5, 6{{\]\]}} output_shape [1, 3, 8, 2, 8, 1, 32] : tensor<1x24x16x32xi8> into tensor<1x3x8x2x8x1x32xi8>
// CHECK: %[[TRANSPOSED0:.+]] = linalg.transpose ins(%[[EXPANDED0]] : tensor<1x3x8x2x8x1x32xi8>) outs([[_:.*]] : tensor<1x3x2x1x8x8x32xi8>) permutation = [0, 1, 3, 5, 2, 4, 6]
// CHECK: %[[PADDED1:.+]] = tensor.pad %[[T1]] low[0, 0, 0, 0] high[0, 4, 0, 18]
// CHECK: %[[EXPANDED1:.+]] = tensor.expand_shape %[[PADDED1]] {{\[\[}}0], [1, 2], [3, 4], [5, 6{{\]\]}} output_shape [1, 3, 8, 2, 8, 1, 32] : tensor<1x24x16x32xi8> into tensor<1x3x8x2x8x1x32xi8>
// CHECK: %[[TRANSPOSED1:.+]] = linalg.transpose ins(%[[EXPANDED1]] : tensor<1x3x8x2x8x1x32xi8>) outs([[_:.*]] : tensor<1x3x2x1x8x8x32xi8>) permutation = [0, 1, 3, 5, 2, 4, 6]
// CHECK: %[[ADDED:.+]] = linalg.add ins(%[[TRANSPOSED0]], %[[TRANSPOSED1]] : tensor<1x3x2x1x8x8x32xi8>, tensor<1x3x2x1x8x8x32xi8>) outs([[_:.*]] : tensor<1x3x2x1x8x8x32xi8>)
// CHECK: %[[TRANSPOSED2:.+]] = linalg.transpose ins(%[[ADDED]] : tensor<1x3x2x1x8x8x32xi8>) outs([[_:.*]] : tensor<1x3x8x2x8x1x32xi8>) permutation = [0, 1, 4, 2, 5, 3, 6]
// CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSED2]] {{\[\[}}0], [1, 2], [3, 4], [5, 6{{\]\]}} : tensor<1x3x8x2x8x1x32xi8> into tensor<1x24x16x32xi8>
// CHECK: %[[UNPADDED:.+]] = tensor.extract_slice %[[COLLAPSED]][0, 0, 0, 0] [1, 20, 16, 14] [1, 1, 1, 1] : tensor<1x24x16x32xi8> to tensor<1x20x16x14xi8>
module {
func.func @main(%in1: memref<1x20x16x14xi8>,
               %in2 : memref<1x20x16x14xi8>,
               %result: memref<1x20x16x14xi8>) {
  %packed_input1 = bufferization.to_tensor %in1 restrict : memref<1x20x16x14xi8> to tensor<1x20x16x14xi8>
  %packed_input2 = bufferization.to_tensor %in2 restrict : memref<1x20x16x14xi8> to tensor<1x20x16x14xi8>
  %c0_i8 = arith.constant 0 : i8
  %e1 = tensor.empty() : tensor<1x3x2x1x8x8x32xi8>
  %packed1 = linalg.pack %packed_input1 padding_value(%c0_i8 : i8) inner_dims_pos = [1,2,3] inner_tiles = [8,8,32] into %e1: tensor<1x20x16x14xi8> -> tensor<1x3x2x1x8x8x32xi8>
  %e2 = tensor.empty() : tensor<1x3x2x1x8x8x32xi8>
  %packed2 = linalg.pack %packed_input2 padding_value(%c0_i8 : i8) inner_dims_pos = [1,2,3] inner_tiles = [8,8,32] into %e2: tensor<1x20x16x14xi8> -> tensor<1x3x2x1x8x8x32xi8>
  %e3 = tensor.empty() : tensor<1x3x2x1x8x8x32xi8>
  %add = linalg.add ins(%packed1, %packed2 : tensor<1x3x2x1x8x8x32xi8>, tensor<1x3x2x1x8x8x32xi8>) outs(%e3: tensor<1x3x2x1x8x8x32xi8>) -> tensor<1x3x2x1x8x8x32xi8>
  %e4 = tensor.empty() : tensor<1x20x16x14xi8>
  %unpacked_out = linalg.unpack %add inner_dims_pos = [1,2,3] inner_tiles = [8,8,32] into %e4 : tensor<1x3x2x1x8x8x32xi8> -> tensor<1x20x16x14xi8>
  bufferization.materialize_in_destination %unpacked_out in writable %result : (tensor<1x20x16x14xi8>, memref<1x20x16x14xi8>) -> ()
  return
}
}
