// RUN: linalg-hexagon-opt %s -hexkl-to-llvm | FileCheck %s

func.func @hexkl_matmul(%arg0: memref<32x64xf16>, %arg1: memref<64x128xf16>, %output: memref<32x128xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<32x64xf16>, memref<64x128xf16>) outs(%output : memref<32x128xf32>)
    return
}

// CHECK-LABEL:    llvm.func @hexkl_matmul_f16f16_f32(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK-LABEL:    func.func @hexkl_matmul
// CHECK:          llvm.call @hexkl_matmul_f16f16_f32
