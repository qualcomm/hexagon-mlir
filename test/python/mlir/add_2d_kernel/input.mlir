func.func @add_2d_mlir(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
  %c0 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [63, 16384], strides: [16384, 1] : memref<*xf32> to memref<63x16384xf32, strided<[16384, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [63, 16384], strides: [16384, 1] : memref<*xf32> to memref<63x16384xf32, strided<[16384, 1], offset: ?>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [63, 16384], strides: [16384, 1] : memref<*xf32> to memref<63x16384xf32, strided<[16384, 1], offset: ?>>
  %alloc = memref.alloc() : memref<63x16384xf32>
  %0 = bufferization.to_tensor %reinterpret_cast restrict : memref<63x16384xf32, strided<[16384, 1], offset: ?>> to tensor<63x16384xf32>
  %alloc_2 = memref.alloc() : memref<63x16384xf32>
  %1 = bufferization.to_tensor %reinterpret_cast_0 restrict : memref<63x16384xf32, strided<[16384, 1], offset: ?>> to tensor<63x16384xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<63x16384xf32>, tensor<63x16384xf32>) outs(%0 : tensor<63x16384xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %3 = arith.addf %in, %in_3 : f32
    linalg.yield %3 : f32
  } -> tensor<63x16384xf32>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_1 : (tensor<63x16384xf32>, memref<63x16384xf32, strided<[16384, 1], offset: ?>>) -> ()
  return
}
