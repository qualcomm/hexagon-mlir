// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(schedule-matmul-for-hvx,linalg-generalize,vtcm-tiling{tile-size-override=64,64,64}), \
// RUN:         one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s -check-prefixes=CHECK

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
func.func @matmul_fp16_tiling_reduction(%x: memref<64x128xf16>, %y: memref<128x64xf16>, %z: memref<64x64xf16>) {
  %t0 = bufferization.to_tensor %x restrict writable : memref<64x128xf16> to tensor<64x128xf16>
  %t1 = bufferization.to_tensor %y restrict writable : memref<128x64xf16> to tensor<128x64xf16>

  %cst = arith.constant 0.000000e+00 : f16
  %t2 = tensor.empty() : tensor<64x64xf16>
  %t3 = linalg.fill ins(%cst : f16) outs(%t2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %t4 = linalg.matmul ins(%t0, %t1 : tensor<64x128xf16>, tensor<128x64xf16>) outs(%t3 : tensor<64x64xf16>) -> tensor<64x64xf16>

  bufferization.materialize_in_destination %t4 in writable %z
      : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
   return
 }
}

// CHECK-LABEL: @matmul_fp16_tiling_reduction
// CHECK-SAME: %[[X:.+]]: memref<64x128xf16>, %[[Y:.+]]: memref<128x64xf16>, %[[Z:.+]]: memref<64x64xf16>
//
// CHECK: %[[ALLOC1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, 1>
// CHECK: memref.copy %{{.*}}, %[[ALLOC1]] : memref<64x64xf16> to memref<64x64xf16, 1>
// CHECK: %[[ALLOC2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, 1>
// CHECK: %[[ALLOC3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, 1>
// CHECK:   scf.for %[[I:.+]] = %c0 to %c128 step %c64 {
// CHECK-DAG:    %[[VX:.+]] = memref.subview %[[X]][0, %[[I]]] [64, 64] [1, 1] : memref<64x128xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
// CHECK-DAG:     memref.copy %[[VX]], %[[ALLOC2]] : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<64x64xf16, 1>
// CHECK-DAG:    %[[VY:.+]] = memref.subview %[[Y]][%[[I]], 0] [64, 64] [1, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
// CHECK-DAG:     memref.copy %[[VY]], %[[ALLOC3]] : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, 1>
// CHECK:        linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : memref<64x64xf16, 1>, memref<64x64xf16, 1>) outs(%{{.*}} : memref<64x64xf16, 1>)
// CHECK: %[[ALLOC4:.+]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16>
// CHECK: memref.copy %[[ALLOC1]], %[[ALLOC4]] : memref<64x64xf16, 1> to memref<64x64xf16>
// CHECK: memref.dealloc %[[ALLOC1]] : memref<64x64xf16, 1>
// CHECK: memref.dealloc %[[ALLOC2]] : memref<64x64xf16, 1>
// CHECK: memref.dealloc %[[ALLOC3]] : memref<64x64xf16, 1>
// CHECK: memref.dealloc %[[ALLOC4]] : memref<64x64xf16>
