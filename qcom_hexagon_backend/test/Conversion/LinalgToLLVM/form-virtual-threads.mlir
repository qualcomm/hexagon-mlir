// RUN: linalg-hexagon-opt %s -form-virtual-threads="block-size=8" | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @kernel(%arg0: memref<1024x2048xf32>, %arg1: memref<1024x2048xf32>, %arg2: memref<1024x2048xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
  %3 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%0, %1 : tensor<1024x2048xf32>, tensor<1024x2048xf32>) outs(%2 : tensor<1024x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x2048xf32>
  bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<1024x2048xf32>, memref<1024x2048xf32>) -> ()
  return
}

// CHECK-LABEL: @kernel
// CHECK-SAME:  (%[[ARG0:.*]]: memref<1024x2048xf32>, %[[ARG1:.*]]: memref<1024x2048xf32>, %[[ARG2:.*]]: memref<1024x2048xf32>)
// CHECK:       %[[T0:.*]] = bufferization.to_tensor %[[ARG0]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[T1:.*]] = bufferization.to_tensor %[[ARG1]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[T2:.*]] = bufferization.to_tensor %[[ARG2]] restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
// CHECK:       %[[FORALL:.*]] = scf.forall (%[[IV:.*]]) = (0) to (1024) step (8) shared_outs(%[[SHARED_OUT:.*]] = %[[T2]]) -> (tensor<1024x2048xf32>) {
// CHECK:         %[[SLICE0:.*]] = tensor.extract_slice %[[T0]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[SLICE1:.*]] = tensor.extract_slice %[[T1]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[SLICE_OUT:.*]] = tensor.extract_slice %[[SHARED_OUT]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<8x2048xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:        ins(%[[SLICE0]], %[[SLICE1]] : tensor<8x2048xf32>, tensor<8x2048xf32>) outs(%[[SLICE_OUT]] : tensor<8x2048xf32>) {
// CHECK:         ^bb0(%[[IN:.*]]: f32, %[[IN2:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:           %[[ADD:.*]] = arith.addf %[[IN]], %[[IN2]] : f32
// CHECK:           linalg.yield %[[ADD]] : f32
// CHECK:         } -> tensor<8x2048xf32>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[GENERIC]] into %[[SHARED_OUT]][%[[IV]], 0] [8, 2048] [1, 1] : tensor<8x2048xf32> into tensor<1024x2048xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       bufferization.materialize_in_destination %[[FORALL]] in writable %[[ARG2]] : (tensor<1024x2048xf32>, memref<1024x2048xf32>) -> ()
