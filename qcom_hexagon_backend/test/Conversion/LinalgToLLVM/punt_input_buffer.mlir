// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-punt-buffer))' -split-input-file | FileCheck %s -check-prefixes=CHECK

// CHECK-LABEL: direct_copy(
// CHECK: %[[ARG0:.*]]: memref<*xf16>,
func.func @direct_copy(
     %arg0: memref<*xf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> f16 {
  %cst = arith.constant 1.419920e+00 : f16
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  %3 = arith.index_cast %arg4 : i32 to index

// CHECK: %[[REINTERPRET_CAST:.+]] = memref.reinterpret_cast %[[ARG0]] to offset
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256, 64], strides: [%1, %2] : memref<*xf16> to memref<256x64xf16, strided<[?, ?], offset: ?>>
  %alloc = memref.alloc() : memref<256x64xf16>

// CHECK-NOT: memref.copy
  memref.copy %reinterpret_cast, %alloc : memref<256x64xf16, strided<[?, ?], offset: ?>> to memref<256x64xf16>

// CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[REINTERPRET_CAST]] restrict : memref<256x64xf16, strided<[?, ?], offset: ?>> to tensor<256x64xf16>
  %4 = bufferization.to_tensor %alloc restrict writable : memref<256x64xf16> to tensor<256x64xf16>

  %5 = tensor.empty() : tensor<256xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
  %reduced = linalg.reduce { arith.addf } ins(%4 : tensor<256x64xf16>) outs(%6 : tensor<256xf16>) dimensions = [1]
  %extracted = tensor.extract %reduced[%3] : tensor<256xf16>
// CHECK: return %{{.+}} : f16
  return %extracted : f16
}

// -----

// CHECK-LABEL: @dont_erase_subview_alloc
func.func @dont_erase_subview_alloc(
     %arg0: memref<*xf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %buffer2: memref<32768xi8>) -> f16 {
  %cst = arith.constant 1.419920e+00 : f16
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  %3 = arith.index_cast %arg4 : i32 to index

  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256, 64], strides: [%1, %2] : memref<*xf16> to memref<256x64xf16, strided<[?, ?], offset: ?>>
  %alloc = memref.view %buffer2[%0][] : memref<32768xi8> to memref<256x64xf16>
// CHECK: memref.copy
  memref.copy %reinterpret_cast, %alloc : memref<256x64xf16, strided<[?, ?], offset: ?>> to memref<256x64xf16>

  %4 = bufferization.to_tensor %alloc restrict writable : memref<256x64xf16> to tensor<256x64xf16>
  %5 = tensor.empty() : tensor<256xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
  %reduced = linalg.reduce { arith.addf } ins(%4 : tensor<256x64xf16>) outs(%6 : tensor<256xf16>) dimensions = [1]
  %extracted = tensor.extract %reduced[%3] : tensor<256xf16>
// CHECK: return %{{.+}} : f16
  return %extracted : f16
}

// -----

// CHECK-LABEL: complex_scf_arg_passing(
// CHECK: %[[ARG0:.*]]: memref<*xf16>,
func.func @complex_scf_arg_passing(%arg0: memref<*xf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> f16 {
  %cst = arith.constant 1.419920e+00 : f16
  %c0_i32 = arith.constant 0 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c64_i32 = arith.constant 64 : i32

  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  %3 = arith.index_cast %arg4 : i32 to index

// CHECK: %[[REINTERPRET_CAST:.+]] = memref.reinterpret_cast %[[ARG0]] to offset
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [64, 64], strides: [%1, %2]
                         : memref<*xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>


// CHECK:  %4:2 = scf.for %arg5 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%[[CUMSUM:.+]] = %{{.*}}, %[[THIS_MEMREF:.+]] = %[[REINTERPRET_CAST:.+]]) -> (f16, memref<64x64xf16, strided<[?, ?], offset: ?>>) : i32 {

  %4:2 = scf.for %arg5 = %c0_i32 to %c1024_i32 step %c64_i32
                iter_args(%arg6 = %cst, %arg7 = %reinterpret_cast)
                -> (f16, memref<64x64xf16, strided<[?, ?], offset: ?>>)  : i32 {

    %alloc = memref.alloc() : memref<64x64xf16>
// CHECK-NOT: memref.copy
    memref.copy %arg7, %alloc : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16>

// CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[THIS_MEMREF]] restrict : memref<64x64xf16, strided<[?, ?], offset: ?>> to tensor<64x64xf16>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16> to tensor<64x64xf16>

    %6 = tensor.empty() : tensor<64xf16>
    %7 = linalg.fill ins(%arg6 : f16) outs(%6 : tensor<64xf16>) -> tensor<64xf16>
    %reduced = linalg.reduce { arith.addf } ins(%5 : tensor<64x64xf16>) outs(%7 : tensor<64xf16>) dimensions = [1]
    %extracted = tensor.extract %reduced[%3] : tensor<64xf16>

    %8 = arith.addi %0, %3 : index

    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [64, 64], strides: [%1, %2] : memref<*xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %extracted, %reinterpret_cast_0 : f16, memref<64x64xf16, strided<[?, ?], offset: ?>>
   }
  return %4#0 : f16
}
