// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Reduce -> Pack -> MMT4D (packed) -> Unpack -> Reduce
//===------------------------------------------------------------------===//

module {
  func.func @reduce_pack_mmt4d_unpack(
      %arg0 : tensor<8x7x5xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<8xi32> {

    // Pre: reduce over the last dim to get A : 8x7
    %a_init = tensor.empty() : tensor<8x7xi32>
    %A = linalg.reduce { arith.maxsi }
           ins(%arg0 : tensor<8x7x5xi32>) outs(%a_init : tensor<8x7xi32>) dimensions = [2]

    // Pack A with inner_tiles = [4, 4] -> 2x2x4x4 (needs padding since 7 % 4 != 0)
    %c0_i32 = arith.constant 0 : i32
    %A_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %A_packed = linalg.pack %A padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %A_pack_dst
      : tensor<8x7xi32> -> tensor<2x2x4x4xi32>

    // Packed 4D matmul must use linalg.mmt4d (not matmul/batch_matmul)
    %C_pack_dst = tensor.empty() : tensor<2x2x4x4xi32>
    %C_packed = linalg.mmt4d
      ins(%A_packed, %argB_packed : tensor<2x2x4x4xi32>, tensor<2x2x4x4xi32>)
      outs(%C_pack_dst : tensor<2x2x4x4xi32>) -> tensor<2x2x4x4xi32>

    // Unpack C back to 8x8
    %C_dst = tensor.empty() : tensor<8x8xi32>
    %C = linalg.unpack %C_packed
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %C_dst
      : tensor<2x2x4x4xi32> -> tensor<8x8xi32>

    // Post: another reduce (same kind)
    %r_dst = tensor.empty() : tensor<8xi32>
    %R = linalg.reduce { arith.maxsi }
           ins(%C : tensor<8x8xi32>) outs(%r_dst : tensor<8xi32>) dimensions = [1]
    return %R : tensor<8xi32>
  }
}

// CHECK: func.func @reduce_pack_mmt4d_unpack(%[[A0:.+]]: tensor<8x7x5xi32>, %[[A1:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [0, 1]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: %[[REDUCE_PRE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[PACK]] : tensor<2x2x5x4x4xi32>
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[REDUCE_PRE]], %[[A1]]
// CHECK: %[[REDUCE_POST:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MMT4D]] : tensor<2x2x4x4xi32>
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[REDUCE_POST]]
// CHECK-SAME: inner_dims_pos = [0] inner_tiles = [4]
// CHECK: return %[[UNPACK]] : tensor<8xi32>
