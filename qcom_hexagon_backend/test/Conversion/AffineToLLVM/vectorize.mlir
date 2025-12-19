// RUN: linalg-hexagon-opt %s -affine-vectorize | FileCheck %s -check-prefixes=CHECK

module attributes {llvm.target_triple = "hexagon"} {
  func.func @elemwise(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>) {

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf16>
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<1024x1024xf16>
        %1 = affine.load %arg1[%arg3, %arg4] : memref<1024x1024xf16>
        %2 = arith.addf %0, %1 : f16
        affine.store %2, %alloc[%arg3, %arg4] : memref<1024x1024xf16>
      }
    }

    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        %0 = affine.load %alloc[%arg3, %arg4] : memref<1024x1024xf16>
        %1 = affine.load %arg1[%arg3, %arg4] : memref<1024x1024xf16>
        %2 = arith.mulf %0, %1 : f16
        affine.store %2, %alloc[%arg3, %arg4] : memref<1024x1024xf16>
      }
    }
    memref.copy %alloc, %arg2 : memref<1024x1024xf16> to memref<1024x1024xf16>
    memref.dealloc %alloc : memref<1024x1024xf16>
    return
  }
}
// CHECK-LABEL: @elemwise
// CHECK:  affine.for %[[K:.+]] = 0 to 1024 {
// CHECK:    affine.for %[[L:.+]] = 0 to 1024 step 32 {
// CHECK:      vector.transfer_read {{.*}} : memref<1024x1024xf16>, vector<32xf16>

// CHECK:  affine.for %[[K:.+]] = 0 to 1024 {
// CHECK:    affine.for %[[L:.+]] = 0 to 1024 step 32 {
// CHECK:      vector.transfer_read {{.*}} : memref<1024x1024xf16>, vector<32xf16>
