module {
  func.func @conv_tiling_hmx(%arg0: memref<1x16x32x128xf16>, %arg1: memref<224x1x1x128xf16>, %arg2: memref<1x16x32x224xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<1x16x32x128xf16> to tensor<1x16x32x128xf16>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<224x1x1x128xf16> to tensor<224x1x1x128xf16>
    %8 = tensor.empty() : tensor<1x16x32x224xf16>
    %9 = linalg.conv_2d_nhwc_fhwc ins(%0, %1 : tensor<1x16x32x128xf16>, tensor<224x1x1x128xf16>) outs(%8 : tensor<1x16x32x224xf16>) -> tensor<1x16x32x224xf16>
    bufferization.materialize_in_destination %9 in restrict writable %arg2 : (tensor<1x16x32x224xf16>, memref<1x16x32x224xf16>) -> ()
    return
  }
}
