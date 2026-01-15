// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false}),linalg-specialize-generic-ops)' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Transpose -> Pack -> MMT4D (packed) -> Unpack -> Transpose Test
//===------------------------------------------------------------------===//

module {

  func.func @transpose_pack_mmt4d_unpack(
      %tin : tensor<8x7xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<8x8xi32> {
    %a_dst = tensor.empty() : tensor<7x8xi32>
    %A = linalg.transpose
           ins(%tin : tensor<8x7xi32>)
           outs(%a_dst : tensor<7x8xi32>) permutation=[1, 0]

    // Pack A with [4,4] -> 2x2x4x4 (needs padding)
    %c0_i32 = arith.constant 0 : i32
    %A_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %A_packed = linalg.pack %A padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %A_pack_dst
      : tensor<7x8xi32> -> tensor<2x2x4x4xi32>

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

    // Post: linalg.transpose after unpack, to propagate through
    %s_dst = tensor.empty() : tensor<8x8xi32>
    %S = linalg.transpose
           ins(%C: tensor<8x8xi32>)
           outs(%s_dst : tensor<8x8xi32>) permutation=[1, 0]
    return %S : tensor<8x8xi32>
  }
}

// CHECK: func.func @transpose_pack_mmt4d_unpack(%[[A0:.+]]: tensor<8x7xi32>, %[[A1:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK_A:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [1, 0]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[TRANSP_A:.*]] = linalg.transpose
// CHECK-SAME: ins(%[[PACK_A]] : tensor<2x2x4x4xi32>)
// CHECK-SAME: permutation = [1, 0, 2, 3]
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[TRANSP_A]], %[[A1]]
// CHECK: %[[TRANSP_POST:.*]] = linalg.transpose
// CHECK-SAME: ins(%[[MMT4D]] : tensor<2x2x4x4xi32>)
// CHECK-SAME: permutation = [1, 0, 2, 3]
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[TRANSP_POST]]
// CHECK-SAME: inner_dims_pos = [1, 0]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: return %[[UNPACK]] : tensor<8x8xi32>
