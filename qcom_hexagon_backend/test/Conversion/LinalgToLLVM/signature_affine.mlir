// RUN: linalg-hexagon-opt %s -linalg-to-llvm  | FileCheck %s

module {
  func.func @store_scalar(%arg0: memref<*xi32>, %given: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
     %c77 = arith.constant 77 : index
    affine.store %given, %reinterpret_cast[%c77] : memref<1024xi32, strided<[1]>>
    return
  }
}

// CHECK: llvm.func @store_scalar[[RANK:.+]]: i64, [[MEMREF_DESC_ADDR:%.+]]: !llvm.ptr, [[VALUE_TO_STORE:%.+]]: i32) {
// CHECK-NEXT: [[ALIGNED_PTR_ADDR:%.+]] = llvm.getelementptr [[MEMREF_DESC_ADDR]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: [[ALIGNED_PTR_TO_BUFFER:%.+]] = llvm.load [[ALIGNED_PTR_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: [[PTR_TO_ELEMENT:%.+]] = llvm.getelementptr inbounds|nuw [[ALIGNED_PTR_TO_BUFFER]][77] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT: llvm.store [[VALUE_TO_STORE]], [[PTR_TO_ELEMENT]] : i32, !llvm.ptr
