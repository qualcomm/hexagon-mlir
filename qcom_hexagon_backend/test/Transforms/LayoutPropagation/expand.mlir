
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// tensor.expand -> Pack -> MMT4D (packed) -> Unpack -> tensor.expand Test
//===------------------------------------------------------------------===//

module {
  func.func @expand_pack_mmt4d_unpack(
      %tin : tensor<64xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<2x4x8xi32> {

    %A = tensor.expand_shape %tin [[0, 1]] output_shape [8, 8] : tensor<64xi32> into tensor<8x8xi32>

    // Pack A with [4,4] -> 2x2x4x4 (needs padding)
    %c0_i32 = arith.constant 0 : i32
    %A_pack_dst0 = tensor.empty() : tensor<8x2x4xi32>
    // linalg.pack can only pack only one dimension that is expanded and
    // only the inner-most expanded dimension should be packed. Otherwise,
    // elements order will be affected after operation reordering.
    %A_packed0 = linalg.pack %A padding_value(%c0_i32 : i32)
      inner_dims_pos = [1] inner_tiles = [4] into %A_pack_dst0
      : tensor<8x8xi32> -> tensor<8x2x4xi32>

    %A_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %A_packed = linalg.pack %A_packed0 padding_value(%c0_i32 : i32)
      inner_dims_pos = [0] inner_tiles = [4] into %A_pack_dst
      : tensor<8x2x4xi32> -> tensor<2x2x4x4xi32>
    
    // Packed 4D matmul -> mmt4d
    %C_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %C_packed = linalg.mmt4d
      ins(%A_packed, %argB_packed : tensor<2x2x4x4xi32>, tensor<2x2x4x4xi32>)
      outs(%C_pack_dst : tensor<2x2x4x4xi32>) -> tensor<2x2x4x4xi32>

    // Unpack C -> 8x8
    %C_dst = tensor.empty() : tensor<8x8xi32>
    %C = linalg.unpack %C_packed
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %C_dst
      : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
    
    %S = tensor.expand_shape %C [[0, 1], [2]] output_shape [2, 4, 8]: tensor<8x8xi32> into tensor<2x4x8xi32>
    return %S : tensor<2x4x8xi32>
  }
}

// CHECK: func.func @expand_pack_mmt4d_unpack(%[[A0:.+]]: tensor<64xi32>, %[[A1:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK_A:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [0]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[PACK_A]]
// CHECK-SAME: output_shape [8, 2, 4] : tensor<16x4xi32> into tensor<8x2x4xi32>
// CHECK: %[[PACK_B:.*]] = linalg.pack %[[EXPAND]]
// CHECK-SAME: inner_dims_pos = [0]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[PACK_B]], %[[A1]]
// CHECK: %[[EXPAND_POST:.*]] = tensor.expand_shape %[[MMT4D]]
// CHECK-SAME: output_shape [2, 1, 2, 4, 4]
// CHECK-SAME: tensor<2x2x4x4xi32> into tensor<2x1x2x4x4xi32>
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[EXPAND_POST]]
// CHECK-SAME: inner_dims_pos = [1, 2]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: return %[[UNPACK]] : tensor<2x4x8xi32>
