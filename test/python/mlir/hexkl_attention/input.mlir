#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @hexkl_attention(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.72134751 : f32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant -1.000000e+14 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1024x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
    %6 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
    %7 = arith.extsi %arg0 : i32 to i64
    %8 = arith.muli %7, %c65536_i64 : i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = arith.muli %arg8, %c1024_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %11, %c64 : index
    %13 = arith.addi %9, %12 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1024x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024x64xf16, strided<[64, 1], offset: ?>> to memref<1024x64xf16>
    %14 = bufferization.to_tensor %alloc restrict writable : memref<1024x64xf16> to tensor<1024x64xf16>
    %15:5 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg12 = %6, %arg13 = %4, %arg14 = %5, %arg15 = %c0_i64, %arg16 = %c0_i64) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, i64, i64)  : i32 {
      %18 = arith.index_cast %arg15 : i64 to index
      %19 = arith.muli %18, %c64 : index
      %20 = arith.addi %9, %19 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [64, 1] : memref<*xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast_5, %alloc_6 : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
      %21 = bufferization.to_tensor %alloc_6 restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %22 = tensor.empty() : tensor<64x64xf16>
      %transposed = linalg.transpose ins(%21 : tensor<64x64xf16>) outs(%22 : tensor<64x64xf16>) permutation = [1, 0] 
      %23 = linalg.matmul ins(%14, %transposed : tensor<1024x64xf16>, tensor<64x64xf16>) outs(%6 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
      %24 = tensor.empty() : tensor<1024x64xf16>
      %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<1024x64xf32>) outs(%24 : tensor<1024x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %51 = arith.truncf %in : f32 to f16
        linalg.yield %51 : f16
      } -> tensor<1024x64xf16>
      %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<1024x64xf16>) outs(%0 : tensor<1024x64xf32>) {
      ^bb0(%in: f16, %out: f32):
        %51 = arith.extf %in : f16 to f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %27 = tensor.empty() : tensor<64x1024xf32>
      %transposed_7 = linalg.transpose ins(%26 : tensor<1024x64xf32>) outs(%27 : tensor<64x1024xf32>) permutation = [1, 0] 
      %28 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
      %reduced = linalg.reduce ins(%transposed_7 : tensor<64x1024xf32>) outs(%28 : tensor<1024xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %51 = arith.maxnumf %in, %init : f32
          linalg.yield %51 : f32
        }
      %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%reduced, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%reduced : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.mulf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %30 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg14, %29 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg14 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.maxnumf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%26, %1 : tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%26 : tensor<1024x64xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.mulf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %expanded_8 = tensor.expand_shape %30 [[0, 1]] output_shape [1024, 1] : tensor<1024xf32> into tensor<1024x1xf32>
      %32 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<1024x1xf32>) outs(%0 : tensor<1024x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1024x64xf32>
      %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%31, %32 : tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%31 : tensor<1024x64xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.subf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<1024x64xf32>) outs(%33 : tensor<1024x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %51 = math.exp2 %in : f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %transposed_9 = linalg.transpose ins(%34 : tensor<1024x64xf32>) outs(%27 : tensor<64x1024xf32>) permutation = [1, 0] 
      %35 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
      %reduced_10 = linalg.reduce ins(%transposed_9 : tensor<64x1024xf32>) outs(%35 : tensor<1024xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %51 = arith.addf %in, %init : f32
          linalg.yield %51 : f32
        }
      %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg14, %30 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg14 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.subf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %37 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%36 : tensor<1024xf32>) outs(%36 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %51 = math.exp2 %in : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %38 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %37 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.mulf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %39 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%38, %reduced_10 : tensor<1024xf32>, tensor<1024xf32>) outs(%38 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.addf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024xf32>
      %expanded_11 = tensor.expand_shape %37 [[0, 1]] output_shape [1024, 1] : tensor<1024xf32> into tensor<1024x1xf32>
      %40 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<1024x1xf32>) outs(%0 : tensor<1024x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1024x64xf32>
      %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %40 : tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%arg12 : tensor<1024x64xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.mulf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %42 = arith.index_cast %arg16 : i64 to index
      %43 = arith.muli %42, %c64 : index
      %44 = arith.addi %9, %43 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [64, 64], strides: [64, 1] : memref<*xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast_12, %alloc_13 : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
      %45 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %46 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<1024x64xf32>) outs(%24 : tensor<1024x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %51 = arith.truncf %in : f32 to f16
        linalg.yield %51 : f16
      } -> tensor<1024x64xf16>
      %47 = linalg.matmul ins(%46, %45 : tensor<1024x64xf16>, tensor<64x64xf16>) outs(%6 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
      %48 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%41, %47 : tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%41 : tensor<1024x64xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %51 = arith.addf %in, %in_14 : f32
        linalg.yield %51 : f32
      } -> tensor<1024x64xf32>
      %49 = arith.addi %arg16, %c64_i64 : i64
      %50 = arith.addi %arg15, %c64_i64 : i64
      scf.yield %48, %39, %30, %50, %49 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, i64, i64
    }
    %expanded = tensor.expand_shape %15#1 [[0, 1]] output_shape [1024, 1] : tensor<1024xf32> into tensor<1024x1xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<1024x1xf32>) outs(%0 : tensor<1024x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x64xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15#0, %16 : tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%15#0 : tensor<1024x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %18 = arith.divf %in, %in_5 : f32
      linalg.yield %18 : f32
    } -> tensor<1024x64xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [1024, 64], strides: [64, 1] : memref<*xf32> to memref<1024x64xf32, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %17 in writable %reinterpret_cast_4 : (tensor<1024x64xf32>, memref<1024x64xf32, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}
