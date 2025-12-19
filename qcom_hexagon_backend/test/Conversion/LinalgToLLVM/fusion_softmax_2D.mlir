// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=true fdmu=true}))' | FileCheck %s -check-prefixes CHECK,RECOMPUTE
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion{far=false fdmu=true}))' | FileCheck %s -check-prefixes CHECK,NO-RECOMPUTE


#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
func.func @fused_softmax_2D(%x : memref<128x128xf32>, %y : memref<128x128xf32>) {
  %X = bufferization.to_tensor %x restrict writable : memref<128x128xf32> to tensor<128x128xf32>
  %Y = bufferization.to_tensor %y restrict writable : memref<128x128xf32> to tensor<128x128xf32>

    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<128x1xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
                     ins(%X : tensor<128x128xf32>) outs(%3 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.maximumf %in, %out : f32
      linalg.yield %13 : f32
    } -> (tensor<128x1xf32>)

    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
                     ins(%X, %4 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%Y : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.subf %in, %in_1 : f32
      linalg.yield %11 : f32
    } -> tensor<128x128xf32>

    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
                     ins(%6 : tensor<128x128xf32>) outs(%Y : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = math.exp %in : f32
      linalg.yield %11 : f32
    } -> tensor<128x128xf32>

    %8 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>

    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
                     ins(%7 : tensor<128x128xf32>) outs(%8 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
    } -> tensor<128x1xf32>

    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
                     ins(%7, %9 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%Y : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.divf %in, %in_1 : f32
      linalg.yield %11 : f32
    } -> tensor<128x128xf32>

    bufferization.materialize_in_destination %10 in writable %y
      : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
   return
 }
}

// CHECK:             [[MAX:%.+]] = arith.maximumf {{.*}}, {{.*}} : f32
// CHECK-NEXT:        linalg.yield [[MAX]] : f32

// COM: When recomputation is allowed, the sequence of:
// COM:	     		   arith.subf ...
// COM:			   math.exp ...
// COM: is computed twice, at the beginning of each of the following linalg.generics
// RECOMPUTE:         [[SUB_0:%.+]] = arith.subf {{.*}}, {{.*}} : f32
// RECOMPUTE-NEXT:    [[EXP_0:%.+]] = math.exp [[SUB_0]] : f32
// RECOMPUTE-NEXT:    [[SUM_0:%.+]] = arith.addf [[EXP_0]], {{.*}} : f32
// RECOMPUTE-NEXT:    linalg.yield [[SUM_0]] : f32

// RECOMPUTE:         [[SUB_1:%.+]] = arith.subf {{.*}}, {{.*}} : f32
// RECOMPUTE-NEXT:    [[EXP_1:%.+]] = math.exp [[SUB_1]] : f32
// RECOMPUTE-NEXT:    [[DIV_0:%.+]] = arith.divf [[EXP_1]], {{.*}} : f32
// RECOMPUTE-NEXT:    linalg.yield [[DIV_0]] : f32

// COM: When recomputation is not allowed, the sequence of:
// COM:	     		   arith.subf ...
// COM:			   math.exp ...
// COM: is computed once, and then directly used in each of the following linalg.generics
// NO-RECOMPUTE:      [[SUB_2:%.+]] = arith.subf {{.*}}, {{.*}} : f32
// NO-RECOMPUTE-NEXT: [[EXP_2:%.+]] = math.exp [[SUB_2]] : f32
// NO-RECOMPUTE-NEXT: linalg.yield [[EXP_2]] : f32

// NO-RECOMPUTE:      [[SUM_1:%.+]] = arith.addf {{.*}}, {{.*}} : f32
// NO-RECOMPUTE-NEXT: linalg.yield [[SUM_1]] : f32

// NO-RECOMPUTE:      [[DIV_1:%.+]] = arith.divf {{.*}}, {{.*}} : f32
// NO-RECOMPUTE-NEXT: linalg.yield [[DIV_1]] : f32
