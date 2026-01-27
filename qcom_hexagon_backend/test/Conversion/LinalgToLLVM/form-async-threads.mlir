// RUN: linalg-hexagon-opt %s -one-shot-bufferize -cse -canonicalize -form-async-threads | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<1024x2048xf32>, %arg1: memref<1024x2048xf32>, %arg2: memref<1024x2048xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<1024x2048xf32> to tensor<1024x2048xf32>
    %3 = scf.forall (%arg3) in (1024) shared_outs(%arg4 = %2) -> (tensor<1024x2048xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg3, 0] [1, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<1x2048xf32>
      %extracted_slice_0 = tensor.extract_slice %1[%arg3, 0] [1, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<1x2048xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [1, 2048] [1, 1] : tensor<1024x2048xf32> to tensor<1x2048xf32>
      %4 = linalg.generic
             {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
             ins(%extracted_slice, %extracted_slice_0 : tensor<1x2048xf32>, tensor<1x2048xf32>)
             outs(%extracted_slice_1 : tensor<1x2048xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %5 = arith.addf %in, %in_2 : f32
        linalg.yield %5 : f32
      } -> tensor<1x2048xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg4[%arg3, 0] [1, 2048] [1, 1] : tensor<1x2048xf32> into tensor<1024x2048xf32>
      }
    }
    bufferization.materialize_in_destination %3 in writable %arg2 : (tensor<1024x2048xf32>, memref<1024x2048xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @kernel(
// CHECK: %[[A0:.+]]: memref<1024x2048xf32>, %[[A1:.+]]: memref<1024x2048xf32>, %[[A2:.+]]: memref<1024x2048xf32>)
// CHECK:   %[[B0:.+]] = arith.constant 0 : index
// CHECK:   %[[B1024:.+]] = arith.constant 1024 : index
// CHECK:   %[[B1:.+]] = arith.constant 1 : index
// CHECK:   %[[N_THREADS:.+]] = arith.divui %[[B1024]], %[[B1]] : index
// CHECK:   %[[G:.+]] = async.create_group %[[N_THREADS]] : !async.group
// CHECK:   scf.for %[[A3:.+]] = %[[B0]] to %[[B1024]] step %[[B1]] {
// CHECK:     %[[T0:.+]] = async.execute {
// CHECK:       %[[S:.+]] = memref.subview %[[A0]][%[[A3]], 0] [1, 2048] [1, 1] : memref<1024x2048xf32> to memref<1x2048xf32, strided<[2048, 1], offset: ?>>
// CHECK:       %[[S1:.+]] = memref.subview %[[A1]][%[[A3]], 0] [1, 2048] [1, 1] : memref<1024x2048xf32> to memref<1x2048xf32, strided<[2048, 1], offset: ?>>
// CHECK:       %[[S2:.+]] = memref.subview %[[A2]][%[[A3]], 0] [1, 2048] [1, 1] : memref<1024x2048xf32> to memref<1x2048xf32, strided<[2048, 1], offset: ?>>
// CHECK:       linalg.generic
// CHECK-SAME:     {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:     ins(%[[S]], %[[S1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>>, memref<1x2048xf32, strided<[2048, 1], offset: ?>>)
// CHECK-SAME:     outs(%[[S2]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>>) {
// CHECK:       ^bb0(%[[I:.+]]: f32, %[[I3:.+]]: f32, %[[O:.+]]: f32):
// CHECK:         %[[A:.+]] = arith.addf %[[I]], %[[I3]] : f32
// CHECK:         linalg.yield %[[A]] : f32
// CHECK:       }
// CHECK:       async.yield
// CHECK:     }
// CHECK:   %[[T1:.+]] = async.add_to_group %[[T0]], %[[G]] : !async.token
// CHECK:   }
// CHECK:   async.await_all %[[G]]
// CHECK:   return
// CHECK: }
// CHECK: }
