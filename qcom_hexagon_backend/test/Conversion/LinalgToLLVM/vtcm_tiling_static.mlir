// RUN: linalg-hexagon-opt -split-input-file %s  -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize, \
// RUN:         one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s -check-prefixes=CHECK

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @static_size_no_tiling(%x: memref<128x256xf32>, %y: memref<128x256xf32>, %z: memref<128x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<128x256xf32> to tensor<128x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<128x256xf32> to tensor<128x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<128x256xf32>, tensor<128x256xf32>)
           outs(%t2 : tensor<128x256xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):

     %4 = arith.addf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<128x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<128x256xf32>, memref<128x256xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @static_size_no_tiling
// CHECK-SAME: %[[X:.+]]: memref<128x256xf32>, %[[Y:.+]]: memref<128x256xf32>, %[[Z:.+]]: memref<128x256xf32>
//
// CHECK: %[[ALLOC1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 1>
// CHECK-NEXT: memref.copy %[[X]], %[[ALLOC1]] : memref<128x256xf32> to memref<128x256xf32, 1>
// CHECK: %[[ALLOC2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 1>
// CHECK-NEXT: memref.copy %[[Y]], %[[ALLOC2]] : memref<128x256xf32> to memref<128x256xf32, 1>
// CHECK: %[[ALLOC3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 1>
// CHECK-NEXT: memref.copy %[[Z]], %[[ALLOC3]] : memref<128x256xf32> to memref<128x256xf32, 1>
// CHECK: linalg.generic {{.*}} ins(%[[ALLOC1]], %[[ALLOC2]] :  memref<128x256xf32, 1>, memref<128x256xf32, 1>) outs(%[[ALLOC3]] : memref<128x256xf32, 1>)
// CHECK: %[[ALLOC4:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32>
// CHECK: memref.copy %[[ALLOC3]], %[[ALLOC4]] :  memref<128x256xf32, 1> to memref<128x256xf32>
// CHECK: memref.dealloc %[[ALLOC1]] : memref<128x256xf32, 1>
// CHECK: memref.dealloc %[[ALLOC2]] : memref<128x256xf32, 1>
// CHECK: memref.dealloc %[[ALLOC3]] : memref<128x256xf32, 1>
// CHECK: memref.dealloc %[[ALLOC4]] : memref<128x256xf32>
//

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
func.func @matmul_fp16_tiling(%x: memref<2048x8192xf16>, %y: memref<8192x1024xf16>, %z: memref<2048x1024xf32>) {
  %t0 = bufferization.to_tensor %x restrict writable : memref<2048x8192xf16> to tensor<2048x8192xf16>
  %t1 = bufferization.to_tensor %y restrict writable : memref<8192x1024xf16> to tensor<8192x1024xf16>
  %t2 = bufferization.to_tensor %z restrict writable : memref<2048x1024xf32> to tensor<2048x1024xf32>

  %t4 = linalg.generic {
          indexing_maps = [#map, #map1, #map2],
          iterator_types = ["parallel", "reduction", "parallel"]}
          ins(%t0, %t1 : tensor<2048x8192xf16>, tensor<8192x1024xf16>)
          outs(%t2 : tensor<2048x1024xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
      %1 = arith.extf %in : f16 to f32
      %2 = arith.extf %in_0 : f16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
  } -> tensor<2048x1024xf32>

  bufferization.materialize_in_destination %t4 in writable %z
      : (tensor<2048x1024xf32>, memref<2048x1024xf32>) -> ()
   return
 }
}

// CHECK-LABEL: @matmul_fp16_tiling
// CHECK-SAME: %[[X:.+]]: memref<2048x8192xf16>, %[[Y:.+]]: memref<8192x1024xf16>, %[[Z:.+]]: memref<2048x1024xf32>
//
// CHECK-3: memref.alloc() {alignment = 64 : i64} : memref<64x4096xf16, 1>
// CHECK-3: memref.alloc() {alignment = 64 : i64} : memref<4096x64xf16, 1>
// CHECK-3: memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, 1>
// CHECK: scf.for %[[I:.+]] = %c0 to %c2048 step %c64 {
// CHECK:   scf.for %[[J:.+]] = %c0 to %c8192 step %c4096 {
// CHECK:     scf.for %[[K:.+]] = %c0 to %c1024 step %c64 {
// CHECK-DAG:    %[[VX:.+]] = memref.subview %[[X]][%[[I]], %[[J]]] [64, 4096] [1, 1] : memref<2048x8192xf16> to memref<64x4096xf16, strided<[8192, 1], offset: ?>>
// CHECK-DAG:     memref.copy %[[VX]], %{{.*}} : memref<64x4096xf16, strided<[8192, 1], offset: ?>> to memref<64x4096xf16, 1>
// CHECK-DAG:    %[[VY:.+]] = memref.subview %[[Y]][%[[J]], %[[K]]] [4096, 64] [1, 1] : memref<8192x1024xf16> to memref<4096x64xf16, strided<[1024, 1], offset: ?>>
// CHECK-DAG:     memref.copy %[[VY]], %{{.*}} : memref<4096x64xf16, strided<[1024, 1], offset: ?>> to memref<4096x64xf16, 1>
// CHECK-DAG:    %[[VZ:.+]] = memref.subview %[[Z]][%[[I]], %[[K]]] [64, 64] [1, 1] : memref<2048x1024xf32> to memref<64x64xf32, strided<[1024, 1], offset: ?>>
// CHECK-DAG:     memref.copy %[[VZ]], %{{.*}} : memref<64x64xf32, strided<[1024, 1], offset: ?>> to memref<64x64xf32, 1>
// CHECK:        linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : memref<64x4096xf16, 1>, memref<4096x64xf16, 1>) outs(%{{.*}} : memref<64x64xf32, 1>)
// CHECK:        memref.copy %{{.*}}, %[[VZ]] : memref<64x64xf32, 1> to memref<64x64xf32, strided<[1024, 1], offset: ?>>
// CHECK-3: memref.dealloc %{{.*}} : memref<64x4096xf16, 1>
// CHECK-3: memref.dealloc %{{.*}} : memref<4096x64xf16, 1>
// CHECK-3: memref.dealloc %{{.*}} : memref<64x64xf32, 1>
