// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s

module {
func.func @kernel(%x: memref<1024xf16>, %y: memref<f16>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<1024xf16> to tensor<1024xf16>
   %t1 = bufferization.to_tensor %y restrict writable : memref<f16> to tensor<f16>

    %reduced = linalg.reduce ins(%t0 : tensor<1024xf16>) outs(%t1 : tensor<f16>) dimensions = [0]
      (%in: f16, %init: f16) {
        %3 = arith.maximumf %in, %init : f16
        linalg.yield %3 : f16
      }
    bufferization.materialize_in_destination %reduced in writable %y
      : (tensor<f16>, memref<f16>) -> ()
   return
 }
}
// CHECK: llvm.intr.vector.reduce.fmaximum(%{{.*}})  {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf16>) -> f16
