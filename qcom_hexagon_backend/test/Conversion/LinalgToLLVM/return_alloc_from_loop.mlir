// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s

// This tests Fix for TritonAI-128.
// There are two parts to this
// 1. alloc return in scf.for loop. Not a great thing to do but some tests end up doing so.
// 2. failure of Bufferization analysis. Fix upstream.

#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @kernel(%X_ptr: memref<f16>, %Y_ptr: memref<f16>, %Z_ptr: memref<f16>, %arg3: i32)  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c12 = arith.constant 12 : index
    %0 = arith.index_cast %arg3 : i32 to index

    %this_X = memref.reinterpret_cast %X_ptr to offset: [%0], sizes: [4, 256], strides: [1, %c5]
                         : memref<f16> to memref<4x256xf16, strided<[1, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x256xf16>
    memref.copy %this_X, %alloc : memref<4x256xf16, strided<[1, ?], offset: ?>> to memref<4x256xf16>
    %X_copy = bufferization.to_tensor %alloc restrict writable : memref<4x256xf16> to tensor<4x256xf16>
    %init_Y = memref.reinterpret_cast %Y_ptr to offset: [%0], sizes: [4, 256], strides: [%c1, %c5]
                             : memref<f16> to memref<4x256xf16, strided<[?, ?], offset: ?>>

    %3:4 = scf.for %arg11 = %c0 to %c12 step %c3
      iter_args(%X = %X_copy, %this_Y = %init_Y, %i = %0, %offset = %c0)
      -> (tensor<4x256xf16>, memref<4x256xf16, strided<[?, ?], offset: ?>>, index, index) {

        %alloc_4 = memref.alloc() : memref<4x256xf16>
        memref.copy %this_Y, %alloc_4 : memref<4x256xf16, strided<[?, ?], offset: ?>> to memref<4x256xf16>
        %Y = bufferization.to_tensor %alloc_4 restrict writable : memref<4x256xf16> to tensor<4x256xf16>

        %X_out = linalg.generic {indexing_maps = [#map, #map, #map],
                             iterator_types = ["parallel", "parallel"]}
                             ins(%X, %Y : tensor<4x256xf16>, tensor<4x256xf16>)
                             outs(%X : tensor<4x256xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %9 = arith.addf %in, %in_6 : f16
          linalg.yield %9 : f16
        } -> tensor<4x256xf16>

        %7 = arith.addi %i, %c3 : index
        %8 = arith.addi %7, %offset : index
        %Y_out = memref.reinterpret_cast %Y_ptr to offset: [%8], sizes: [4, 256], strides: [%c1, %c5]
                    : memref<f16> to memref<4x256xf16, strided<[?, ?], offset: ?>>
        scf.yield %X_out, %Y_out, %8, %c0
                    : tensor<4x256xf16>, memref<4x256xf16, strided<[?, ?], offset: ?>>, index, index
    }
    %reinterpret_cast_3 = memref.reinterpret_cast %Z_ptr to offset: [%0], sizes: [4, 256], strides: [1, %c5]
                            : memref<f16> to memref<4x256xf16, strided<[1, ?], offset: ?>>
    bufferization.materialize_in_destination %3#0 in writable %reinterpret_cast_3
                            : (tensor<4x256xf16>, memref<4x256xf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}
// CHECK: module attributes {llvm.target_triple = "hexagon"} {
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr
// CHECK: llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
// CHECK: llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
// CHECK-SAME: %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64,
// CHECK-SAME: %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64,
// CHECK-SAME: %arg9: i32) {
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: yield
// CHECK: llvm.return
