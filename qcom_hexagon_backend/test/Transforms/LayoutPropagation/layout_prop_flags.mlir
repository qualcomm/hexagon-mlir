// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=true}))' | FileCheck %s --check-prefixes=CHECK_PAR_ONLY
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{parallels-only=false}))' | FileCheck %s --check-prefixes=CHECK_PAR_RED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{upper=false}))' | FileCheck %s --check-prefixes=CHECK_NOT_UPPER
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-generalize,extend-pack-frontier{lower=false}))' | FileCheck %s --check-prefixes=CHECK_NOT_LOWER
//===------------------------------------------------------------------===//
// lit tests showing the layout propagation flag expected behavior:
// --parallels-only (default=true), --lower (default=true), and --upper (default=true)
//===------------------------------------------------------------------===//

  func.func @parallels_only(
      %arg0 : tensor<8x7x5xi32>,
      %arg1 : tensor<8x7xi32>,
      %arg2 : tensor<8x8xi32>) -> tensor<8xi32> {

    // Pre: reduce over the last dim to get A : 8x7
    %0 = tensor.empty() : tensor<8x7xi32>
    %reduce = linalg.reduce { arith.maxsi }
           ins(%arg0 : tensor<8x7x5xi32>) outs(%0 : tensor<8x7xi32>) dimensions = [2]

    %1 = tensor.empty() : tensor<8x7xi32>
    %sub = linalg.sub
           ins(%reduce, %arg1 : tensor<8x7xi32>, tensor<8x7xi32>)
           outs(%1 : tensor<8x7xi32>) -> tensor<8x7xi32>

    // Pack A with inner_tiles = [4, 4] -> 2x2x4x4 (needs padding since 7 % 4 != 0)
    %c0_i32 = arith.constant 0 : i32
    %2 = tensor.empty() : tensor<2x2x4x4xi32>
    %pack = linalg.pack %sub padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %2
      : tensor<8x7xi32> -> tensor<2x2x4x4xi32>
    
    %3 = tensor.empty() : tensor<8x8xi32>
    %unpack = linalg.unpack %pack 
      inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %3
      : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
    %4 = tensor.empty() : tensor<8x8xi32>
    %sub_post = linalg.sub
           ins(%unpack, %arg2 : tensor<8x8xi32>, tensor<8x8xi32>)
           outs(%4 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %5 = tensor.empty() : tensor<8xi32>
    %reduce_post = linalg.reduce { arith.maxsi }
           ins(%sub_post : tensor<8x8xi32>) outs(%5 : tensor<8xi32>) dimensions = [1]
    return %reduce_post : tensor<8xi32>
}

// CHECK_PAR_ONLY: linalg.generic
// CHECK_PAR_ONLY-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK_PAR_ONLY: linalg.pack
// CHECK_PAR_ONLY: linalg.pack
// CHECK_PAR_ONLY: linalg.generic
// CHECK_PAR_ONLY-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_PAR_ONLY: linalg.generic
// CHECK_PAR_ONLY-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_PAR_ONLY: linalg.unpack
// CHECK_PAR_ONLY: linalg.generic
// CHECK_PAR_ONLY-SAME: iterator_types = ["parallel", "reduction"]

// CHECK_PAR_RED: linalg.pack
// CHECK_PAR_RED: linalg.generic
// CHECK_PAR_RED-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"] 
// CHECK_PAR_RED: linalg.pack
// CHECK_PAR_RED: linalg.generic
// CHECK_PAR_RED-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_PAR_RED: linalg.generic
// CHECK_PAR_RED-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_PAR_RED: linalg.generic
// CHECK_PAR_RED-SAME: iterator_types = ["parallel", "reduction", "parallel", "reduction"]
// CHECK_PAR_RED: linalg.unpack

// CHECK_NOT_UPPER: linalg.generic
// CHECK_NOT_UPPER-SAME: iterator_types = ["parallel", "parallel", "reduction"] 
// CHECK_NOT_UPPER: linalg.generic
// CHECK_NOT_UPPER-SAME: iterator_types = ["parallel", "parallel"]
// CHECK_NOT_UPPER: linalg.pack
// CHECK_NOT_UPPER: linalg.generic
// CHECK_NOT_UPPER-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_NOT_UPPER: linalg.unpack
// CHECK_NOT_UPPER: linalg.generic
// CHECK_NOT_UPPER-SAME: iterator_types = ["parallel", "reduction"]

// CHECK_NOT_LOWER: linalg.generic
// CHECK_NOT_LOWER-SAME: iterator_types = ["parallel", "parallel", "reduction"] 
// CHECK_NOT_LOWER: linalg.pack
// CHECK_NOT_LOWER: linalg.pack
// CHECK_NOT_LOWER: linalg.generic
// CHECK_NOT_LOWER-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK_NOT_LOWER: linalg.unpack
// CHECK_NOT_LOWER: linalg.generic
// CHECK_NOT_LOWER-SAME: iterator_types = ["parallel", "parallel"]
// CHECK_NOT_LOWER: linalg.generic
// CHECK_NOT_LOWER-SAME: iterator_types = ["parallel", "reduction"]
