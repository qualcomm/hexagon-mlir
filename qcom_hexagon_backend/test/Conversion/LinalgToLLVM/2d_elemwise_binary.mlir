// RUN: linalg-hexagon-opt %s -linalg-to-llvm | \
// RUN: linalg-hexagon-translate -emit=llvmir  | FileCheck %s

// This test checks the llvm-ir strictly to ensure that basic 2-D tensor sub is
// done efficiently no matter how other opts behave.
// Not all tests have to be this restricted.

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
// CHECK-LABEL: @kernel(ptr readnone captures(none) %0, ptr %1, i64 %2,
// CHECK-SAME:          i64 %3, i64 %4, i64 %5, i64 %6, ptr readnone captures(none) %7, ptr %8,
// CHECK:      tail call void @hexagon_runtime_copy(ptr [[VTCM_X:%.+]], ptr %1, i32 524288, i1 true, i1 false)
// CHECK:      tail call void @hexagon_runtime_copy(ptr [[VTCM_Y:%.+]], ptr %8, i32 524288, i1 true, i1 false)
// CHECK:      {{.*}} = phi
// CHECK-NEXT: {{.*}} = shl
// CHECK-NEXT: {{.*}} = trunc
// CHECK-NEXT: [[GEP_X:%.+]] = getelementptr float, ptr [[VTCM_X]]
// CHECK-NEXT: [[LOAD_X:%.+]] = load <32 x float>, ptr [[GEP_X]]
// CHECK-NEXT: [[GEP_Y:%.+]] = getelementptr float, ptr [[VTCM_Y]]
// CHECK-NEXT: [[LOAD_Y:%.+]] = load <32 x float>, ptr [[GEP_Y]]
// CHECK-NEXT: [[SUB:%.+]] = fsub {{(fast)?}} <32 x float> [[LOAD_X]], [[LOAD_Y]]
// CHECK-NEXT: [[GEP_Z:%.+]]  = getelementptr float, ptr
// CHECK-NEXT: store <32 x float> [[SUB]], ptr [[GEP_Z]]
