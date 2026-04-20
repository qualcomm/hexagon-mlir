
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK
//===------------------------------------------------------------------===//
// lit test for assumptions and limitations for the layout propagation pass
//===------------------------------------------------------------------===//

// The checks for lit tests in this file should be replaced with lit tests verifying feature functionality
// The goal of these lit tests is to keep track of assumptions/missing features when developers expect a given
// layout propagation case to work.

// This example demonstrates the GENERIC_PAD_SUPPORT limitation described in ExtendPackUnpackFrontier.cpp
// Propagation of linalg.pack with padding through generics that don't contain divides are documented in
// sub.mlir and reduce.mlir
func.func @no_prop_through_div_generic(%arg: tensor<11xf32>) -> tensor<6x2xf32> {
  %res = tensor.empty() : tensor<11xf32>
  %div = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, // for %arg
      affine_map<(d0) -> (d0)> // for %res
    ],
    iterator_types = ["parallel"]
  } ins(%arg : tensor<11xf32>)
    outs(%res : tensor<11xf32>) {
  ^bb0(%arg_one: f32, %res_one: f32):
    %c2 = arith.constant 2.0 : f32
    %div = arith.divf %arg_one, %c2 : f32
    linalg.yield %div : f32
  } -> tensor<11xf32>

  %c0_i32 = arith.constant 0.0 : f32
  %dest = tensor.empty() : tensor<6x2xf32>
  %packed = linalg.pack %div padding_value(%c0_i32 : f32)
    inner_dims_pos = [0]
    inner_tiles = [2]
    into %dest : tensor<11xf32> -> tensor<6x2xf32>

  return %packed: tensor<6x2xf32>
}

// CHECK: func.func @no_prop_through_div_generic(%[[A0:.+]]: tensor<11xf32>)
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK: %[[PACK:.*]] = linalg.pack %[[GENERIC]]
// CHECK: return %[[PACK:.*]]

//This example demonstrates the PAD_OF_PACKED_DIM limitation

func.func @pad_of_packed_dim_limitation(
  %tin : tensor<2x8x8xi32>) -> tensor<2x2x8x2x4xi32> {

  %c0_i32 = arith.constant 0 : i32
  %A = tensor.pad %tin low[1, 0, 0] high[1, 0, 0] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
    tensor.yield %c0_i32 : i32
  } : tensor<2x8x8xi32> to tensor<4x8x8xi32>

  %A_pack_dst = tensor.empty() : tensor<2x2x8x2x4xi32>
  %A_packed = linalg.pack %A
    inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %A_pack_dst
    : tensor<4x8x8xi32> -> tensor<2x2x8x2x4xi32>
  return %A_packed : tensor<2x2x8x2x4xi32>
}

// CHECK: func.func @pad_of_packed_dim_limitation(%[[A0:.+]]: tensor<2x8x8xi32>)
// CHECK: %[[PADDED:.*]] = tensor.pad
// CHECK: %[[PACK:.*]] = linalg.pack %[[PADDED]]
// CHECK: return %[[PACK:.*]]

//This example demonstrates the PACK_WITH_PAD_THROUGH_PAD limitation

func.func @pad_of_padded_pack_limitation(
    %tin : tensor<2x7x8xi32>) -> tensor<4x4x2x2x4xi32> {

  %c0_i32 = arith.constant 0 : i32
  %A = tensor.pad %tin low[1, 0, 0] high[1, 0, 0] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
    tensor.yield %c0_i32 : i32
  } : tensor<2x7x8xi32> to tensor<4x7x8xi32>

  %A_pack_dst = tensor.empty() : tensor<4x4x2x2x4xi32>
  //We are packing different dimensions than padded dimension with
  //tensor.pad to avoid triggering the PAD_OF_PACKED_DIM limitation
  //and test PACK_WITH_PAD_THROUGH_PAD in isolation
  %A_packed = linalg.pack %A padding_value(%c0_i32 : i32)
    inner_dims_pos = [1, 2] inner_tiles = [2, 4] into %A_pack_dst
    : tensor<4x7x8xi32> -> tensor<4x4x2x2x4xi32>
  return %A_packed : tensor<4x4x2x2x4xi32>
}

// CHECK: func.func @pad_of_padded_pack_limitation(%[[A0:.+]]: tensor<2x7x8xi32>)
// CHECK: %[[PADDED:.*]] = tensor.pad
// CHECK: %[[PACK:.*]] = linalg.pack %[[PADDED]]
// CHECK: return %[[PACK:.*]]


//This example demonstrates the PACK_OUTPUT_TO_EMPTY_4PAD limitation
func.func @pack_output_to_non_empty_tensor(
    %tin0 : tensor<2x8x8xi32>, %tin1 : tensor<4x4x2x2x4xi32>) -> tensor<4x4x2x2x4xi32> {

  %c0_i32 = arith.constant 0 : i32
  %A = tensor.pad %tin0 low[1, 0, 0] high[1, 0, 0] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
    tensor.yield %c0_i32 : i32
  } : tensor<2x8x8xi32> to tensor<4x8x8xi32>

  %A_packed = linalg.pack %A
    inner_dims_pos = [1, 2] inner_tiles = [2, 4] into %tin1
    : tensor<4x8x8xi32> -> tensor<4x4x2x2x4xi32>
  return %A_packed : tensor<4x4x2x2x4xi32>
}

// CHECK: func.func @pack_output_to_non_empty_tensor(%[[A0:.+]]: tensor<2x8x8xi32>,
// CHECK-SAME: %[[A1:.+]]: tensor<4x4x2x2x4xi32>)
// CHECK: %[[PADDED:.*]] = tensor.pad
// CHECK: %[[PACK:.*]] = linalg.pack %[[PADDED]]
// CHECK-SAME: into %[[A1:.+]] : tensor<4x8x8xi32>
// CHECK: return %[[PACK:.*]]

// This example demonstrates the UNPACK_INNER_SIZES_DIV_PROJECTED_DIMS limitation

func.func @unpack_inner_sizes_div_proj_dims(%arg0 : tensor<2x2x4x4xi32>) -> tensor<2x4x8xi32> {
  %0 = tensor.empty() : tensor<8x8xi32>  
  %unpacked = linalg.unpack %arg0
    inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %0
    : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
  // dim 0 is expanded into [0, 1] -> unpacking on dim 0 can be projected to unpack on dim 1
  //the projected dims are [1, 2]
  %expanded = tensor.expand_shape %unpacked [[0, 1], [2]] output_shape [2, 4, 8]: tensor<8x8xi32> into tensor<2x4x8xi32>
  return %expanded : tensor<2x4x8xi32>
}
