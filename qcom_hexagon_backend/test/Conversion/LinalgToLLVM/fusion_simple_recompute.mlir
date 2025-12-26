// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=false fdmu=false}))' | FileCheck %s -check-prefix FAR_OFF_FDMU_OFF
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=true fdmu=false}))' | FileCheck %s -check-prefix FAR_ON_FDMU_OFF
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=true fdmu=true}))' | FileCheck %s -check-prefix FAR_ON_FDMU_ON
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=false fdmu=true}))' | FileCheck %s -check-prefix FAR_OFF_FDMU_ON

#map = affine_map<(d0) -> (d0)>
module {
func.func @kernel(%x: memref<1024xf32>, %y: memref<1024xf32>,
		  %a: memref<1024xf32>, %b: memref<1024xf32>,
		  %out_a: memref<1024xf32>, %out_b: memref<1024xf32>) {
    %X = bufferization.to_tensor %x restrict writable : memref<1024xf32> to tensor<1024xf32>
    %Y = bufferization.to_tensor %y restrict writable : memref<1024xf32> to tensor<1024xf32>
    %A = bufferization.to_tensor %a restrict writable : memref<1024xf32> to tensor<1024xf32>
    %B = bufferization.to_tensor %b restrict writable : memref<1024xf32> to tensor<1024xf32>

    %eltwise_add = linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
            ins(%X, %Y : tensor<1024xf32>, tensor<1024xf32>)
            outs(%X : tensor<1024xf32>) {
    ^bb0(%in_0: f32, %in_1: f32, %out: f32):
      %prod = arith.mulf %in_0, %in_1 : f32
      linalg.yield %prod : f32
    } -> tensor<1024xf32>

    %tmp_a = linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
            ins(%eltwise_add, %A : tensor<1024xf32>, tensor<1024xf32>)
            outs(%A : tensor<1024xf32>) {
    ^bb0(%in_0: f32, %in_1: f32, %out: f32):
      %quot = arith.divf %in_0, %in_1 : f32
      linalg.yield %quot : f32
    } -> tensor<1024xf32>

    %tmp_b = linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
            ins(%eltwise_add, %B : tensor<1024xf32>, tensor<1024xf32>)
            outs(%B : tensor<1024xf32>) {
    ^bb0(%in_0: f32, %in_1: f32, %out: f32):
      %quot = arith.divf %in_0, %in_1 : f32
      linalg.yield %quot : f32
    } -> tensor<1024xf32>

    bufferization.materialize_in_destination %tmp_a in writable %out_a
					     : (tensor<1024xf32>, memref<1024xf32>) -> ()

    bufferization.materialize_in_destination %tmp_b in writable %out_b
					     : (tensor<1024xf32>, memref<1024xf32>) -> ()

    return
}}

// COM: Uses the config: {fusion=true far=false fdmu=false}
// COM: No fusion is done here (the first 2 linalg.generics are a simple, viable candidate for fusable)
// FAR_OFF_FDMU_OFF:      [[MUL:%.+]] = arith.mulf {{.*}}, {{.*}} : f32
// FAR_OFF_FDMU_OFF-NEXT: linalg.yield [[MUL]] : f32

// COM: Uses the config: {fusion=true far=true fdmu=false}
// COM: Fuses the mulf and first divf but misses the 2nd divf
// FAR_ON_FDMU_OFF:       [[MUL:%.+]] = arith.mulf {{.*}}, {{.*}} : f32
// FAR_ON_FDMU_OFF-NEXT:  [[DIV:%.+]] = arith.divf [[MUL]], {{.*}} : f32
// FAR_ON_FDMU_OFF-NEXT:  linalg.yield [[DIV]] : f32

// COM: Uses the config: {fusion=true far=true fdmu=true}
// COM: Identical results to the above config: {fusion=true far=true fdmu=false}
// FAR_ON_FDMU_ON:        [[MUL:%.+]] = arith.mulf {{.*}}, {{.*}} : f32
// FAR_ON_FDMU_ON-NEXT:   [[DIV:%.+]] = arith.divf [[MUL]], {{.*}} : f32
// FAR_ON_FDMU_ON-NEXT:   linalg.yield [[DIV]] : f32

// COM: Uses the config: {fusion=true far=false fdmu=true}
// COM: Fuses all operations into one linalg.generic with multiple outputs
// COM: (For the given example, this is the ideal result)
// FAR_OFF_FDMU_ON:       [[MUL:%.+]] = arith.mulf {{.*}}, {{.*}} : f32
// FAR_OFF_FDMU_ON-NEXT:  [[DIV0:%.+]] = arith.divf [[MUL]], {{.*}} : f32
// FAR_OFF_FDMU_ON-NEXT:  [[DIV1:%.+]] = arith.divf [[MUL]], {{.*}} : f32
// FAR_OFF_FDMU_ON-NEXT:  linalg.yield [[DIV0]], [[DIV1]] : f32, f32
