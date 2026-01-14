// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hexkl))' | FileCheck %s

func.func @matmul_hexkl(%arg0: memref<256x128xf16>,
                  %arg1: memref<128x64xf16>,
                  %output: memref<256x64xf32>) {
  %arg0_tensor = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16>
  %arg1_tensor = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
  %empty = tensor.empty() : tensor<256x64xf32>
  %res = linalg.matmul ins (%arg0_tensor, %arg1_tensor: tensor<256x128xf16>, tensor<128x64xf16>)
                     outs (%empty: tensor<256x64xf32>) -> tensor<256x64xf32>
  bufferization.materialize_in_destination %res in restrict writable %output
                                : (tensor<256x64xf32>, memref<256x64xf32>) -> ()
  return
}

// CHECK-LABEL: @matmul_hexkl
// CHECK:  %[[Input1:.*]] = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16>
// CHECK:  %[[Input2:.*]] = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
// CHECK:  %[[Empty:.*]] = tensor.empty() : tensor<256x64xf32>
// CHECK:  %[[Res:.*]] = hexkl.matmul ins(%[[Input1]], %[[Input2]] : tensor<256x128xf16>, tensor<128x64xf16>) outs(%[[Empty]] : tensor<256x64xf32>) -> tensor<256x64xf32>
// CHECK:  bufferization.materialize_in_destination %[[Res]] in restrict writable %arg2 : (tensor<256x64xf32>, memref<256x64xf32>) -> ()
