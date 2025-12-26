// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(schedule-matmul-for-hvx))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)> 
module attributes {llvm.target_triple = "hexagon"} {
  // CHECK-LABEL: @fp16_matmul
  // CHECK-SAME: (%arg0: tensor<1024x64xf16>, %arg1: tensor<64x1024xf16>, %arg2: tensor<1024x1024xf32>)
  // CHECK: %0 = linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]}
  // CHECK-SAME: ins(%arg0, %arg1 : tensor<1024x64xf16>, tensor<64x1024xf16>) outs(%arg2 : tensor<1024x1024xf32>)

  func.func @fp16_matmul(%arg0: tensor<1024x64xf16>, %arg1: tensor<64x1024xf16>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %4 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x64xf16>, tensor<64x1024xf16>) outs(%arg2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %4 : tensor<1024x1024xf32>
  }

  // CHECK-LABEL: func.func @matmul(
  // CHECK-SAME: %arg0: tensor<64x256xf16>,
  // CHECK-SAME: %arg1: tensor<256x128xf16>,
  // CHECK-SAME: %arg2: tensor<64x128xf16>) -> tensor<64x128xf16> {
  // CHECK: %0 = linalg.matmul {library_call = "foo"}
  // CHECK: return %0 : tensor<64x128xf16>
  func.func @matmul(%arg0: tensor<64x256xf16>,
                  %arg1: tensor<256x128xf16>,
                  %arg2: tensor<64x128xf16>) -> tensor<64x128xf16> {
    %0 = linalg.matmul {library_call = "foo"} ins (%arg0, %arg1: tensor<64x256xf16>, tensor<256x128xf16>)
                outs (%arg2: tensor<64x128xf16>) -> tensor<64x128xf16>
    return %0 : tensor<64x128xf16>
  }
}
