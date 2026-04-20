// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(force-hvx-crouton, extend-pack-frontier{upper=false}))' | FileCheck %s
module {    
    func.func @gelu_bptr_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 5.000000e-01 : f16
    %cst_0 = arith.constant 7.978520e-01 : f16
    %cst_1 = arith.constant 3.567500e-02 : f16
    %cst_2 = arith.constant 1.000000e+00 : f16
    %cst_3 = arith.constant -2.000000e+00 : f16
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 64, 128, 2], strides: [16384, 256, 2, 1] : memref<*xf16> to memref<64x64x128x2xf16, strided<[16384, 256, 2, 1]>>
    %0 = tensor.empty() : tensor<64x64x128x2xf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 64, 128, 2], strides: [16384, 256, 2, 1] : memref<*xf16> to memref<64x64x128x2xf16, strided<[16384, 256, 2, 1]>>
    %1 = bufferization.to_tensor %reinterpret_cast_4 restrict : memref<64x64x128x2xf16, strided<[16384, 256, 2, 1]>> to tensor<64x64x128x2xf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<64x64x128x2xf16>) outs(%0 : tensor<64x64x128x2xf16>) {
    ^bb0(%in: f16, %out: f16):
        %3 = arith.mulf %in, %cst_1 : f16
        %4 = arith.mulf %3, %in : f16
        %5 = arith.mulf %4, %in : f16
        %6 = arith.mulf %in, %cst_0 : f16
        %7 = arith.addf %6, %5 : f16
        %8 = arith.mulf %7, %cst_3 : f16
        %9 = arith.extf %8 : f16 to f32
        %10 = math.exp %9 : f32
        %11 = arith.truncf %10 : f32 to f16
        %12 = arith.addf %11, %cst_2 : f16
        %13 = arith.subf %cst_2, %11 : f16
        %14 = arith.extf %13 : f16 to f32
        %15 = arith.extf %12 : f16 to f32
        %16 = arith.divf %14, %15 : f32
        %17 = arith.truncf %16 : f32 to f16
        %18 = arith.addf %17, %cst_2 : f16
        %19 = arith.mulf %in, %cst : f16
        %20 = arith.mulf %19, %18 : f16
        linalg.yield %20 : f16
    } -> tensor<64x64x128x2xf16>
    bufferization.materialize_in_destination %2 in restrict writable %reinterpret_cast : (tensor<64x64x128x2xf16>, memref<64x64x128x2xf16, strided<[16384, 256, 2, 1]>>) -> ()
    return
    }
}
// CHECK-LABEL: func.func @gelu_bptr_kernel

// --- pack ops ---
// CHECK: linalg.pack
// CHECK: linalg.pack

// --- main generic op ---
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]

// CHECK: ^bb0(
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: math.exp
// CHECK: linalg.yield

// --- unpack ops ---
// CHECK: linalg.unpack
// CHECK-SAME: inner_dims_pos = [5]
// CHECK-SAME: inner_tiles = [2]

// CHECK: linalg.unpack
// CHECK-SAME: inner_dims_pos = [1, 2, 3]
// CHECK-SAME: inner_tiles = [8, 4, 32]

