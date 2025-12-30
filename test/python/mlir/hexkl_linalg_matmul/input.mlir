module {
  func.func @hexkl_linalg_matmul(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<1024x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024x64xf16, strided<[64, 1]>> to memref<1024x64xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024x64xf16> to tensor<1024x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
    %alloc_1 = memref.alloc() : memref<64x512xf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x512xf16, strided<[512, 1]>> to memref<64x512xf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x512xf16> to tensor<64x512xf16>
    %2 = tensor.empty() : tensor<1024x512xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<1024x64xf16>, tensor<64x512xf16>) outs(%3 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf32> to memref<1024x512xf32, strided<[512, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<1024x512xf32>, memref<1024x512xf32, strided<[512, 1]>>) -> ()
    return
  }
}
