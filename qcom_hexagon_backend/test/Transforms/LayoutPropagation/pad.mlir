// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// tensor.pad -> Pack -> MMT4D (packed) -> Unpack -> tensor.pad Test
//===------------------------------------------------------------------===//

module {
  func.func @pad_pack_mmt4d_unpack(
      %tin : tensor<2x8x8xi32>,
      %argB_packed : tensor<8x2x4x4xi32>) -> tensor<10x32x4xi32> {

    %c0_i32 = arith.constant 0 : i32
    %A = tensor.pad %tin low[1, 0, 0] high[1, 0, 0] {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      tensor.yield %c0_i32 : i32
    } : tensor<2x8x8xi32> to tensor<4x8x8xi32>

    %A_pack_dst = tensor.empty() : tensor<4x2x2x4x4xi32>
    %A_packed = linalg.pack %A
      inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %A_pack_dst
      : tensor<4x8x8xi32> -> tensor<4x2x2x4x4xi32>
    // Collapse is needed to be able to use linalg.mmt4d, since the latter works on 4D tensors
    // only instead of 5D
    %A_packed_collapsed = tensor.collapse_shape %A_packed [[0, 1], [2], [3], [4]]
    : tensor<4x2x2x4x4xi32> into tensor<8x2x4x4xi32>

    // Packed 4D matmul -> mmt4d
    %C_pack_dst = tensor.empty() : tensor<8x8x4x4xi32>
    %C_packed = linalg.mmt4d
      ins(%A_packed_collapsed, %argB_packed : tensor<8x2x4x4xi32>, tensor<8x2x4x4xi32>)
      outs(%C_pack_dst : tensor<8x8x4x4xi32>) -> tensor<8x8x4x4xi32>

    // Unpack C -> 8x8
    %C_dst = tensor.empty() : tensor<8x32x4xi32>
    %C = linalg.unpack %C_packed
      inner_dims_pos = [1] inner_tiles = [4] into %C_dst
      : tensor<8x8x4x4xi32> -> tensor<8x32x4xi32>

    %S = tensor.pad %C low[1, 0, 0] high[1, 0, 0] {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      tensor.yield %c0_i32 : i32
    } : tensor<8x32x4xi32> to tensor<10x32x4xi32>
    
    return %S : tensor<10x32x4xi32>
  }
}

// CHECK: func.func @pad_pack_mmt4d_unpack(%[[A0:.+]]: tensor<2x8x8xi32>, %[[A1:.+]]: tensor<8x2x4x4xi32>)
// CHECK: %[[PACK:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [1, 2]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[PAD:.*]] = tensor.pad %[[PACK]] low[1, 0, 0, 0, 0] high[1, 0, 0, 0, 0]
// CHECK: %[[COLLAPS:.*]] = tensor.collapse_shape %[[PAD]]
// CHECK-SAME: tensor<4x2x2x4x4xi32> into tensor<8x2x4x4xi32>
// Pack can propagate through pad only if padded dimension was not packed 
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[COLLAPS]], %[[A1]]
// CHECK: %[[PAD2:.*]] = tensor.pad %[[MMT4D]]
// Unpack can propagate through pad only if padded dimension was not unpacked 
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[PAD2]]
// CHECK-SAME: inner_dims_pos = [1]
// CHECK-SAME: inner_tiles = [4]
// CHECK: return %[[UNPACK]] : tensor<10x32x4xi32>
