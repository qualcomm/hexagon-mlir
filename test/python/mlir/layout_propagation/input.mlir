// The unpack_then_reduce function is an adaptation of the function in the
// qcom_hexagon_backend/test/Transforms/LayoutPropagation/reduce_generics.mlir
// lit test, and the function reduce_then_unpack is the output of running
// layout propagation through the reduction generic

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @unpack_then_reduce(%arg0_memref: memref<2x2x4x4xi32>, %arg1_memref: memref<8xi32>) -> () {
  %c0 = arith.constant 0 : i32 

  %arg0 = bufferization.to_tensor %arg0_memref restrict: memref<2x2x4x4xi32> to tensor<2x2x4x4xi32>
  %5 = tensor.empty() : tensor<8x8xi32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %5 : tensor<2x2x4x4xi32> -> tensor<8x8xi32>
  %6 = tensor.generate {
  ^bb0(%i : index):
    tensor.yield %c0 : i32 
  } : tensor<8xi32>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%unpack : tensor<8x8xi32>) outs(%6 : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %8 = arith.maxsi %out, %in : i32
    linalg.yield %8 : i32
  } -> tensor<8xi32>
  bufferization.materialize_in_destination %7 in restrict writable %arg1_memref
                                : (tensor<8xi32>, memref<8xi32>) -> ()
  return
}

#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
func.func @reduce_then_unpack(%arg0: memref<2x2x4x4xi32>, %arg1: memref<8xi32>) {
  %c0 = arith.constant 0 : i32 
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x2x4x4xi32> to tensor<2x2x4x4xi32>
  %1 = tensor.empty() : tensor<8xi32>
  %2 = tensor.generate {
  ^bb0(%i : index, %j : index):
    tensor.yield %c0 : i32 
  } : tensor<2x4xi32>

  %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins(%0 : tensor<2x2x4x4xi32>) outs(%2 : tensor<2x4xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.maxsi %out, %in : i32
    linalg.yield %4 : i32
  } -> tensor<2x4xi32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0] inner_tiles = [4] into %1 : tensor<2x4xi32> -> tensor<8xi32>
  bufferization.materialize_in_destination %unpack in restrict writable %arg1 : (tensor<8xi32>, memref<8xi32>) -> ()
  return
}
