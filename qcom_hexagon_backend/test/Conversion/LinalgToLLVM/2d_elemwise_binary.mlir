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
// This generic is pure parallel + identity, so VTCM staging is deliberately
// skipped (VTCMTiling.cpp: "streaming"): it must stream straight through DDR
// with vectorized loads/stores, and never pay the extra VTCM round trip.
// CHECK-LABEL: @kernel(ptr readnone captures(none) %0, ptr readonly captures(none) %1, i64 %2,
// CHECK-SAME:          i64 %3, i64 %4, i64 %5, i64 %6, ptr readnone captures(none) %7, ptr readonly captures(none) %8,
// CHECK-NOT:  hexagon_runtime_copy_dsp
// CHECK:      [[GEP_X:%.+]] = getelementptr [4 x i8], ptr %1
// CHECK-NEXT: [[LOAD_X:%.+]] = load <32 x float>, ptr [[GEP_X]], align 4
// CHECK-NEXT: [[GEP_Y:%.+]] = getelementptr [4 x i8], ptr %8
// CHECK-NEXT: [[LOAD_Y:%.+]] = load <32 x float>, ptr [[GEP_Y]], align 4
// CHECK-NEXT: [[SUB:%.+]] = fsub fast <32 x float> [[LOAD_X]], [[LOAD_Y]]
// CHECK-NEXT: [[GEP_Z:%.+]]  = getelementptr [4 x i8], ptr
// CHECK-NEXT: store <32 x float> [[SUB]], ptr [[GEP_Z]]
