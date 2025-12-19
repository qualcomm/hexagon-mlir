// RUN: linalg-hexagon-opt %s -linalg-to-llvm="enable-multi-threading=true"  | \
// RUN: linalg-hexagon-translate -emit=llvmir  | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @kernel(%x: memref<1024x256xf32>, %y: memref<1024x256xf32>, %z: memref<1024x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<1024x256xf32>, tensor<1024x256xf32>)
           outs(%t2 : tensor<1024x256xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):

     %4 = arith.subf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<1024x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<1024x256xf32>, memref<1024x256xf32>) -> ()
   return
 }
}

// CHECK:   %{{.*}} = tail call ptr @mlirAsyncRuntimeCreateGroup
// CHECK:   %{{.*}} = tail call ptr @mlirAsyncRuntimeCreateToken
// CHECK:   store ptr @async_execute_fn.resume, ptr %{{.*}}
// CHECK:   store ptr @async_execute_fn.destroy, ptr %{{.*}}
// CHECK:   tail call void @mlirAsyncRuntimeExecute(ptr nonnull %{{.*}},
// CHECK:   %{{.*}} = tail call i64 @mlirAsyncRuntimeAddTokenToGroup(ptr %{{.*}}, ptr %{{.*}})
