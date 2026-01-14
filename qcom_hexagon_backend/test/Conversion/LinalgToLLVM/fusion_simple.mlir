// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion))' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
func.func @kernel(%x: memref<1024xf32>, %offset : f32, %z: memref<1024xf32>) {
   %X = bufferization.to_tensor %x restrict writable : memref<1024xf32> to tensor<1024xf32>
   %Z = bufferization.to_tensor %z restrict writable : memref<1024xf32> to tensor<1024xf32>

    %t_empty= tensor.empty() : tensor<1024xf32>
    %t0 = linalg.fill ins(%offset : f32) outs(%t_empty : tensor<1024xf32>) -> tensor<1024xf32>
    %t1 = linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
            ins(%X, %t0 : tensor<1024xf32>, tensor<1024xf32>)
            outs(%X : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %21 = arith.subf %in, %in_8 : f32
      linalg.yield %21 : f32
    } -> tensor<1024xf32>

    %tmp = linalg.generic
            {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
            ins(%t1 : tensor<1024xf32>)
            outs(%Z : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %21 = math.exp %in : f32
      linalg.yield %21 : f32
    } -> tensor<1024xf32>

   bufferization.materialize_in_destination %tmp in writable %z
      : (tensor<1024xf32>, memref<1024xf32>) -> ()
   return
 }
}

// CHECK: [[SUB:%.+]] = arith.subf {{.*}}, {{.*}} : f32
// CHECK-NEXT: [[EXP:%.+]] = math.exp [[SUB]] : f32
// CHECK-NEXT: linalg.yield [[EXP]] : f32
