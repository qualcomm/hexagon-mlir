
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// tensor.expand -> Pack -> MMT4D (packed) -> Unpack -> tensor.expand Test
//===------------------------------------------------------------------===//

module {
  func.func @collapse_pack_mmt4d_unpack(
      %tin : tensor<2x4x8xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<2x32xi32> {

    %A = tensor.collapse_shape %tin [[0, 1], [2]] : tensor<2x4x8xi32> into tensor<8x8xi32>

    // Pack A with [4,4] -> 2x2x4x4 (needs padding)
    // linalg.pack can pack only one dimension that is collapsed and only the inner-most expanded dimension 
    // should be packed, if we want layout propagation to be effective. Otherwise, elements order will be
    // affected after operation reordering.
    %c0_i32 = arith.constant 0 : i32
    %A_pack_dst0 = tensor.empty() : tensor<8x2x4xi32>
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
    %C_dst = tensor.empty() : tensor<2x8x4xi32>
    %C = linalg.unpack %C_packed
      inner_dims_pos = [1] inner_tiles = [4] into %C_dst
      : tensor<2x2x4x4xi32> -> tensor<2x8x4xi32>

    %S = tensor.collapse_shape %C [[0], [1, 2]] : tensor<2x8x4xi32> into tensor<2x32xi32>
    return %S : tensor<2x32xi32>
  }
}

// CHECK: func.func @collapse_pack_mmt4d_unpack(%[[A0:.+]]: tensor<2x4x8xi32>, %[[A1:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK_A:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [2]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[PACK_B:.*]] = linalg.pack %[[PACK_A]]
// CHECK-SAME: inner_dims_pos = [1]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[PACK_B]]
// CHECK-SAME: tensor<2x1x2x4x4xi32> into tensor<2x2x4x4xi32>
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[COLLAPSED]], %[[A1]]
// Layout propagation through collapse_shape is not implemented
// The following CHECKs will need to be changed if we end up
// implementing unpack propagation through collapse_shape
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[MMT4D]]
// CHECK-SAME: inner_dims_pos = [1]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[COLLAPSE_POST:.*]] = tensor.collapse_shape %[[UNPACK]]
// CHECK-SAME: tensor<2x8x4xi32> into tensor<2x32xi32>
// CHECK: return %[[COLLAPSE_POST]] : tensor<2x32xi32>
