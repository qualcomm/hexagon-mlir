// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm | FileCheck %s

#map = affine_map < (d0)->(d0)>
module {
  func.func @test_f16_1D(%arg0: memref<*xf16>, %arg1: memref<1024xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf16> to memref<1024xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<1024xf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024xf16, strided<[1]>> to memref<1024xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xf16> to tensor<1024xf16>
    %1 = tensor.empty() : tensor<1024xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf16>) outs(%1 : tensor<1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<1024xf32>
    bufferization.materialize_in_destination %2 in writable %arg1
      : (tensor<1024xf32>, memref<1024xf32>) -> ()
    return
  }
}
// CHECK-LABEL:  llvm.func @test_f16_1D(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64) {
// CHECK: {{.*}} = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

// -----

#map = affine_map < (d0, d1)->(d0, d1)>
module {
  func.func @test_f32_2D(%arg0 : memref<*xf32>, %arg1 : memref<128x128xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset : [0], sizes : [ 128, 128 ], strides : [ 1, 1 ] : memref<*xf32> to memref<128x128xf32, strided<[ 1, 1 ]>>
    %alloc = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x128xf32, strided<[ 1, 1 ]>> to memref<128x128xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32> to tensor<128x128xf32>

    %result = linalg.generic{indexing_maps = [ #map, #map ],
                                  iterator_types =
                                      [ "parallel",
                                        "parallel" ]}
                     ins(%0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
      ^bb0(%in : f32, %out : f32) :
        %11 = math.exp %in : f32
        linalg.yield %11 : f32
    }
    ->tensor<128x128xf32>
    bufferization.materialize_in_destination %result in writable %arg1
      : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
    return
  }
}
// CHECK-LABEL: llvm.func @test_f32_2D(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) {
