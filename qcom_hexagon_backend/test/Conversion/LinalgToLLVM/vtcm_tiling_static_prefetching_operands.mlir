// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(vtcm-tiling{tile-size-override=7716,128,256}), \
// RUN:         one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s -check-prefixes=CHECK

#mapWZ = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mapX = affine_map<(d0, d1, d2) -> (d1, d2)>
#mapY = affine_map<(d0, d1, d2) -> (d1)>
module {
func.func @tileWithXYfullyInVTCM(
               %W : memref<123456x128x256xf32>, 
               %X: memref<128x256xf32>,
               %Y: memref<128xf32>,
              %Z: memref<123456x128x256xf32>) {
   %tW = bufferization.to_tensor %W restrict writable : memref<123456x128x256xf32> to tensor<123456x128x256xf32>
   %tX = bufferization.to_tensor %X restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %tY = bufferization.to_tensor %Y restrict writable : memref<128xf32> to tensor<128xf32>
   %tZ = bufferization.to_tensor %Z restrict writable : memref<123456x128x256xf32> to tensor<123456x128x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#mapWZ, #mapX, #mapY, #mapWZ],
           iterator_types = ["parallel", "parallel", "parallel"]}
           ins(%tW, %tX, %tY : tensor<123456x128x256xf32>, tensor<128x256xf32>, tensor<128xf32>)
           outs(%tZ : tensor<123456x128x256xf32>) {
   ^bb0(%w: f32, %x : f32,  %y: f32, %z: f32):
     %wx = arith.addf %w, %x : f32
     %wxy = arith.addf %wx, %y : f32
     %wxyz = arith.addf %wxy, %z : f32
     linalg.yield %wxyz : f32
   } -> tensor<123456x128x256xf32>

   bufferization.materialize_in_destination %t3 in writable %Z
      : (tensor<123456x128x256xf32>, memref<123456x128x256xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @tileWithXYfullyInVTCM
// CHECK-SAME: %[[W:.+]]: memref<123456x128x256xf32>, %[[X:.+]]: memref<128x256xf32>, %[[Y:.+]]: memref<128xf32>, %[[Z:.+]]: memref<123456x128x256xf32>
//
// CHECK: %[[ALLOC1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 1>
// CHECK: memref.copy %{{.*}}, %[[ALLOC1]] : memref<128x256xf32> to memref<128x256xf32, 1>
// CHECK: %[[ALLOC2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
// CHECK: memref.copy %{{.*}}, %[[ALLOC2]] : memref<128xf32> to memref<128xf32, 1>
// CHECK: %[[ALLOC3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<7716x128x256xf32, 1>
// CHECK-NEXT: %[[ALLOC4:.+]] = memref.alloc() {alignment = 64 : i64} : memref<7716x128x256xf32, 1>
// CHECK: scf.for %[[I:.+]] = %c0 to %c123456 step %c7716 {
// CHECK-DAG: %[[VW:.+]] = memref.subview %[[W]][%[[I]], 0, 0] [7716, 128, 256] [1, 1, 1] : memref<123456x128x256xf32> to memref<7716x128x256xf32, strided<[32768, 256, 1], offset: ?>>
// CHECK-DAG: memref.copy %[[VW]], %[[ALLOC3]] : memref<7716x128x256xf32, strided<[32768, 256, 1], offset: ?>> to memref<7716x128x256xf32, 1>
// CHECK-DAG: %[[VZ:.+]] = memref.subview %[[Z]][%[[I]], 0, 0] [7716, 128, 256] [1, 1, 1] : memref<123456x128x256xf32> to memref<7716x128x256xf32, strided<[32768, 256, 1], offset: ?>>
// CHECK-DAG: memref.copy %[[VZ]], %[[ALLOC4]] : memref<7716x128x256xf32, strided<[32768, 256, 1], offset: ?>> to memref<7716x128x256xf32, 1>
// CHECK:     linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : memref<7716x128x256xf32, 1>, memref<128x256xf32, 1>, memref<128xf32, 1>) outs(%{{.*}} : memref<7716x128x256xf32, 1>)
// CHECK:     memref.copy %[[ALLOC4]], %[[VZ]] : memref<7716x128x256xf32, 1> to memref<7716x128x256xf32, strided<[32768, 256, 1], offset: ?>>
// CHECK: memref.dealloc %{{.*}} : memref<128x256xf32, 1>
// CHECK: memref.dealloc %{{.*}} : memref<128xf32, 1>
// CHECK: memref.dealloc %{{.*}} : memref<7716x128x256xf32, 1>
// CHECK: memref.dealloc %{{.*}} : memref<7716x128x256xf32, 1>
