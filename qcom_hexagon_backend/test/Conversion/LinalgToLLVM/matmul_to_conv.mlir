// RUN: linalg-hexagon-opt %s -matmul-to-conv | FileCheck %s

// CHECK-LABEL: @matmul
// CHECK-NEXT:  %[[A:.*]] = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16> 
// CHECK-NEXT:  %[[B:.*]] = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
// CHECK-NEXT:  %[[Empty:.*]] = tensor.empty() : tensor<256x64xf16>
// CHECK-NEXT:  %[[Empty2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK-NEXT:  %[[Transposed:.*]] = linalg.transpose ins(%[[B]] : tensor<128x64xf16>) outs(%[[Empty2]] : tensor<64x128xf16>) permutation = [1, 0] 
// CHECK-NEXT:  %[[Expanded:.*]] = tensor.expand_shape %[[A]] {{\[}}[0, 1, 2], [3]] output_shape [1, 8, 32, 128] : tensor<256x128xf16> into tensor<1x8x32x128xf16>
// CHECK-NEXT:  %[[Expanded0:.*]] = tensor.expand_shape %[[Transposed]] {{\[}}[0], [1, 2, 3]] output_shape [64, 1, 1, 128] : tensor<64x128xf16> into tensor<64x1x1x128xf16>
// CHECK-NEXT:  %[[Expanded1:.*]] = tensor.expand_shape %[[Empty]] {{\[}}[0, 1, 2], [3]] output_shape [1, 8, 32, 64] : tensor<256x64xf16> into tensor<1x8x32x64xf16>
// CHECK-NEXT:  %[[Conv:.*]] = linalg.conv_2d_nhwc_fhwc ins(%[[Expanded]], %[[Expanded0]] : tensor<1x8x32x128xf16>, tensor<64x1x1x128xf16>) outs(%[[Expanded1]] : tensor<1x8x32x64xf16>) -> tensor<1x8x32x64xf16>
// CHECK-NEXT:  %[[Collapsed:.*]] = tensor.collapse_shape %[[Conv]] {{\[}}[0, 1, 2], [3]] : tensor<1x8x32x64xf16> into tensor<256x64xf16>
// CHECK-NEXT:  bufferization.materialize_in_destination %[[Collapsed]] in restrict writable %arg2 : (tensor<256x64xf16>, memref<256x64xf16>) -> ()
// CHECK-NEXT:  return
func.func @matmul(%arg0: memref<256x128xf16>,
                  %arg1: memref<128x64xf16>,
                  %output: memref<256x64xf16>) {
  %arg0_tensor = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16>
  %arg1_tensor = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
  %empty = tensor.empty() : tensor<256x64xf16>
  %res = linalg.matmul ins (%arg0_tensor, %arg1_tensor: tensor<256x128xf16>, tensor<128x64xf16>)
                     outs (%empty: tensor<256x64xf16>) -> tensor<256x64xf16>
  bufferization.materialize_in_destination %res in restrict writable %output
                                : (tensor<256x64xf16>, memref<256x64xf16>) -> ()
  return
}
