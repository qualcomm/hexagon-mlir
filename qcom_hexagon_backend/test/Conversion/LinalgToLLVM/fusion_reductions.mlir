// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion))' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
module {
func.func @kernel(%x: memref<1024xf32>, %offset : f32, %z: memref<1024xf32>) {
    %X = bufferization.to_tensor %x restrict writable : memref<1024xf32> to tensor<1024xf32>
    %cst = arith.constant 0xFF800000 : f32
    %t_empty= tensor.empty() : tensor<1xf32>
    %t_min = linalg.fill ins(%offset : f32) outs(%t_empty : tensor<1xf32>) -> tensor<1xf32>
    %max = linalg.generic
            {indexing_maps = [#map, #map1], iterator_types = ["reduction"]}
            ins(%X : tensor<1024xf32>)
            outs(%t_min : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tmp_max = arith.maximumf %in, %out : f32
      linalg.yield %tmp_max : f32
    } -> tensor<1xf32>

    %val_plus_max = linalg.generic
            {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]}
            ins(%X, %max : tensor<1024xf32>, tensor<1xf32>)
            outs(%X : tensor<1024xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %sum = arith.addf %in0, %in1 : f32
      linalg.yield %sum : f32
    } -> tensor<1024xf32>

   bufferization.materialize_in_destination %val_plus_max in writable %z
      : (tensor<1024xf32>, memref<1024xf32>) -> ()
   return
 }
}

// COM: Assert that the linalg.generics don't fuse
// CHECK-COUNT-2: linalg.generic
