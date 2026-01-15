
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false},linalg-specialize-generic-ops))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// Broadcast -> Pack -> MMT4D (packed) -> Unpack -> Broadcast
//===------------------------------------------------------------------===//
module {
  func.func @broadcast_pack_mmt4d_unpack(
      %arg0 : tensor<8xi32>,
      %argB_packed : tensor<2x2x4x4xi32>) -> tensor<8x8x2xi32> {
    // Pre: broadcast to get A : 8x7
    %a_dst = tensor.empty() : tensor<8x7xi32>
    %A = linalg.broadcast
           ins(%arg0 : tensor<8xi32>) outs(%a_dst : tensor<8x7xi32>) dimensions = [1]

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
    
    // Post: broadcast again (same kind)
    %b_dst = tensor.empty() : tensor<8x8x2xi32>
    %Cb = linalg.broadcast
            ins(%C : tensor<8x8xi32>) outs(%b_dst : tensor<8x8x2xi32>) dimensions = [2]
    return %Cb : tensor<8x8x2xi32>
  }
}
// CHECK: func.func @broadcast_pack_mmt4d_unpack(%[[A0:.+]]: tensor<8xi32>, %[[A1:.+]]: tensor<2x2x4x4xi32>)
// CHECK: %[[PACK:.*]] = linalg.pack %[[A0]]
// CHECK-SAME: inner_dims_pos = [0]
// CHECK-SAME: inner_tiles = [4]
// CHECK: %[[BROADCAST1:.*]] = linalg.broadcast
// CHECK-SAME: ins(%[[PACK]] : tensor<2x4xi32>)
// CHECK-SAME: dimensions = [1, 3]
// CHECK: %[[MMT4D:.*]] = linalg.mmt4d
// CHECK-SAME: ins(%[[BROADCAST1]], %[[A1]]
// CHECK: %[[BROADCAST2:.*]] = linalg.broadcast
// CHECK-SAME: ins(%[[MMT4D]] : tensor<2x2x4x4xi32>)
// CHECK-SAME: dimensions = [2]
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[BROADCAST2]]
// CHECK-SAME: inner_dims_pos = [0, 1]
// CHECK-SAME: inner_tiles = [4, 4]
// CHECK: return %[[UNPACK]] : tensor<8x8x2xi32>
