// RUN: linalg-hexagon-opt --eliminate-redundant-unpack-pack %s | FileCheck %s

// Test that redundant unpack/pack pairs are eliminated when the unpack's result
// is consumed by a pack operation.
// The pattern is: input -> unpack1 -> unpack2 -> pack1 -> pack2 -> output
// After optimization, this should simplify to: input -> output (identity)

func.func @eliminate_unpack_pack_chain(%arg0: tensor<1x1x2x24x8x2x32x2xf16>) -> tensor<1x1x2x24x8x2x32x2xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  
  // First unpack: 8D -> 7D (unpack inner dim 5 with tile size 2)
  %0 = tensor.empty() : tensor<1x1x2x24x8x4x32xf16>
  %unpack1 = linalg.unpack %arg0 inner_dims_pos = [5] inner_tiles = [2] 
             into %0 : tensor<1x1x2x24x8x2x32x2xf16> -> tensor<1x1x2x24x8x4x32xf16>
  
  // Second unpack: 7D -> 4D (unpack inner dims 1,2,3 with tile sizes 8,4,32)
  %1 = tensor.empty() : tensor<1x1x7x768xf16>
  %unpack2 = linalg.unpack %unpack1 inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] 
             into %1 : tensor<1x1x2x24x8x4x32xf16> -> tensor<1x1x7x768xf16>
  
  // First pack: 4D -> 7D (pack inner dims 1,2,3 with tile sizes 8,4,32)
  %2 = tensor.empty() : tensor<1x1x2x24x8x4x32xf16>
  %pack1 = linalg.pack %unpack2 padding_value(%cst : f16) 
           inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] 
           into %2 : tensor<1x1x7x768xf16> -> tensor<1x1x2x24x8x4x32xf16>
  
  // Second pack: 7D -> 8D (pack inner dim 5 with tile size 2)
  %3 = tensor.empty() : tensor<1x1x2x24x8x2x32x2xf16>
  %pack2 = linalg.pack %pack1 inner_dims_pos = [5] inner_tiles = [2] 
           into %3 : tensor<1x1x2x24x8x4x32xf16> -> tensor<1x1x2x24x8x2x32x2xf16>
  
  return %pack2 : tensor<1x1x2x24x8x2x32x2xf16>
}

// CHECK-LABEL: func.func @eliminate_unpack_pack_chain
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x1x2x24x8x2x32x2xf16>) -> tensor<1x1x2x24x8x2x32x2xf16>
// CHECK-NEXT:    return %[[ARG0]] : tensor<1x1x2x24x8x2x32x2xf16>
// CHECK-NEXT:  }

// -----

// Negative test: unpack's source type doesn't match pack's dest type
// The optimization should NOT be applied because the tensor types are incompatible.
// unpack source: tensor<1x1x2x24x8x2x32x2xf16> (8D)
// pack dest:     tensor<1x1x2x24x8x1x32x4xf16> (8D, but different dimensions 5 and 7)

func.func @no_eliminate_type_mismatch(%arg0: tensor<1x1x2x24x8x2x32x2xf16>) -> tensor<1x1x2x24x8x1x32x4xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  
  // Unpack: 8D -> 7D (unpack inner dim 5 with tile size 2)
  %0 = tensor.empty() : tensor<1x1x2x24x8x4x32xf16>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [5] inner_tiles = [2] 
            into %0 : tensor<1x1x2x24x8x2x32x2xf16> -> tensor<1x1x2x24x8x4x32xf16>
  
  // Pack: 7D -> 8D (pack inner dim 5 with tile size 4, different from unpack!)
  // Note: pack dest type has dims [5,7] = [1,4], while unpack source has [5,7] = [2,2]
  %1 = tensor.empty() : tensor<1x1x2x24x8x1x32x4xf16>
  %pack = linalg.pack %unpack padding_value(%cst : f16) 
          inner_dims_pos = [5] inner_tiles = [4] 
          into %1 : tensor<1x1x2x24x8x4x32xf16> -> tensor<1x1x2x24x8x1x32x4xf16>
  
  return %pack : tensor<1x1x2x24x8x1x32x4xf16>
}

// CHECK-LABEL: func.func @no_eliminate_type_mismatch
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x1x2x24x8x2x32x2xf16>) -> tensor<1x1x2x24x8x1x32x4xf16>
// CHECK:         %[[EMPTY0:.*]] = tensor.empty() : tensor<1x1x2x24x8x4x32xf16>
// CHECK:         %[[UNPACK:.*]] = linalg.unpack %[[ARG0]]
// CHECK:         %[[EMPTY1:.*]] = tensor.empty() : tensor<1x1x2x24x8x1x32x4xf16>
// CHECK:         %[[PACK:.*]] = linalg.pack %[[UNPACK]]
// CHECK:         return %[[PACK]] : tensor<1x1x2x24x8x1x32x4xf16>
// CHECK:       }
