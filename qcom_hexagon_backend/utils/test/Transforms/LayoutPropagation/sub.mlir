// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Sub -> Pack -> MMT4D (packed) -> Unpack -> Sub Test
//===------------------------------------------------------------------===//

module {
  func.func @sub_pack_mmt4d_unpack(
      %lhs : tensor<8x7xi32>,
      %rhs : tensor<8x7xi32>,
      %post : tensor<8x8xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<8x8xi32> {
    // Pre: sub to get A : 8x7
    %a_dst = tensor.empty() : tensor<8x7xi32>
    %A = linalg.sub
           ins(%lhs, %rhs : tensor<8x7xi32>, tensor<8x7xi32>)
           outs(%a_dst : tensor<8x7xi32>) -> tensor<8x7xi32>

    // Pack A with [4,4] -> 2x2x4x4 (needs padding)
    %c0_i32 = arith.constant 0 : i32
    %A_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %A_packed = linalg.pack %A padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %A_pack_dst
      : tensor<8x7xi32> -> tensor<2x2x4x4xi32>

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

    // Post: sub again (same kind)
    %s_dst = tensor.empty() : tensor<8x8xi32>
    %S = linalg.sub
           ins(%C, %post : tensor<8x8xi32>, tensor<8x8xi32>)
           outs(%s_dst : tensor<8x8xi32>) -> tensor<8x8xi32>
    return %S : tensor<8x8xi32>
  }
}

// CHECK: func.func @sub_pack_mmt4d_unpack(%[[A0:.+]]: tensor<8x7xi32>, %[[A1:.+]]: tensor<8x7xi32>, %[[A2:.+]]: tensor<8x8xi32>, %[[A3:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK_A:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [0, 1]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[PACK_B:.*]] = linalg.pack %[[A1]]
// CHECK-SAME: inner_dims_pos = [0, 1]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[SUB_AB:.*]] = linalg.generic
// CHECK-SAME: ins(%[[PACK_A]], %[[PACK_B]] : tensor<2x2x4x4xi32>, tensor<2x2x4x4xi32>)
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[SUB_AB]], %[[A3]]

// One of the inputs didn't come from an unpack therefore it needs
// a pack instruction since the new linalg.sub will operate on packed data
// CHECK: %[[PACK_C:.*]] = linalg.pack %[[A2]]
// CHECK-SAME: inner_dims_pos = [0, 1]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[SUB_MC:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MMT4D]], %[[PACK_C]] : tensor<2x2x4x4xi32>, tensor<2x2x4x4xi32>)
