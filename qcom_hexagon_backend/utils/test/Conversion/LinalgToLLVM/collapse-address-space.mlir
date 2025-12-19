// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm="enable-vtcm-tiling=true enable-convert-to-hexagonmem=true enable-hexagonmem-copy-to-dma=true" \
// RUN:  | FileCheck %s -check-prefixes=ON

// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   -linalg-to-llvm="enable-vtcm-tiling=true enable-convert-to-hexagonmem=true enable-collapse-address-space=false" \
// RUN:  | FileCheck %s -check-prefixes=OFF

// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm | FileCheck %s --check-prefixes=CHECK

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @foo(%x: memref<2048x1024xf32>, %y: memref<2048x1024xf32>, %z: memref<2048x1024xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<2048x1024xf32> to tensor<2048x1024xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<2048x1024xf32> to tensor<2048x1024xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<2048x1024xf32> to tensor<2048x1024xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<2048x1024xf32>, tensor<2048x1024xf32>)
           outs(%t2 : tensor<2048x1024xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):
     %4 = arith.subf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<2048x1024xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<2048x1024xf32>, memref<2048x1024xf32>) -> ()
   return
 }
}
// ON: llvm.call @hexagon_runtime_dma_start
// ON-NOT: builtin.unrealized_conversion_cast
//
// OFF: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<{{.*}}, 1>
// OFF-NEXT: builtin.unrealized_conversion_cast %{{.*}} : memref<{{.*}}, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>

// -----

module {
 func.func @hexmem_copy(
        %arg0 : memref<2x32x32xf32, strided<[1024, 32, 1], offset: 0>>,
        %result0 : memref<2x32x32xf32, strided<[1024, 32, 1], offset: 0>, 1>,
        %ind: index) {
    hexagonmem.copy %arg0, %result0 
        : memref<2x32x32xf32, strided<[1024, 32, 1], offset: 0>> 
        to memref<2x32x32xf32, strided<[1024, 32, 1], offset: 0>, 1>
    return
 }
}
// CHECK-LABEL: llvm.func @hexmem_copy
// CHECK: %[[SRC_PTR:.*]] = llvm.getelementptr %arg1[{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[DST_PTR:.*]] = llvm.getelementptr %arg10[{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.return
