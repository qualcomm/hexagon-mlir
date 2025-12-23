// RUN: linalg-hexagon-opt -split-input-file %s -pass-pipeline='builtin.module(func.func(convert-to-hexagonmem \
// RUN: ))' | FileCheck %s -check-prefixes=CHECK,ON
//
#map = affine_map < (d0, d1)->(d0, d1)>
module {
  func.func @static_size(%arg0 : memref<2048x1024xf32>, %arg1 : memref<2048x1024xf32>,
                         %arg2 : memref<2048x1024xf32>) {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x1024xf32, 1>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x1024xf32, 1>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<128x1024xf32, 1>
    scf.for %arg3 = %c0 to %c2048 step %c128 {
      %subview = memref.subview %arg0[%arg3, 0] [128, 1024] [1, 1]
                   : memref<2048x1024xf32> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
      memref.copy %subview, %alloc
                   : memref<128x1024xf32, strided<[1024, 1], offset: ?>> to memref<128x1024xf32, 1>
      %subview_2 = memref.subview %arg1[%arg3, 0] [128, 1024] [1, 1]
                   : memref<2048x1024xf32> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
      memref.copy %subview_2, %alloc_0
                   : memref<128x1024xf32, strided<[1024, 1], offset: ?>> to memref<128x1024xf32, 1>
      %subview_3 = memref.subview %arg2[%arg3, 0] [128, 1024] [1, 1]
                   : memref<2048x1024xf32> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
      memref.copy %subview_3, %alloc_1
                   : memref<128x1024xf32, strided<[1024, 1], offset: ?>> to memref<128x1024xf32, 1>
      linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%alloc, %alloc_0 : memref<128x1024xf32, 1>, memref<128x1024xf32, 1>)
            outs(%alloc_1 : memref<128x1024xf32, 1>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %0 = arith.subf %in, %in_4 : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_1, %subview_3
               : memref<128x1024xf32, 1> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
    }
    memref.dealloc %alloc : memref<128x1024xf32, 1>
    memref.dealloc %alloc_0 : memref<128x1024xf32, 1>
    memref.dealloc %alloc_1 : memref<128x1024xf32, 1>
    return
  }
}
// CHECK-LABEL: @static_size
// ON-3: hexagonmem.alloc() : memref<128x1024xf32, 1>
//
// CHECK: scf.for %arg3 = %c0 to %c2048 step %c128
// ON-3:    hexagonmem.copy {{.*}}, {{.*}} : memref<128x1024xf32, strided<[1024,1], offset: ?>> to memref<128x1024xf32, 1>
//
// ON:      hexagonmem.copy {{.*}}, {{.*}} : memref<128x1024xf32, 1> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
//
// ON-3:   hexagonmem.dealloc %0 : memref<128x1024xf32, 1>

// -----

// CHECK-LABEL: @convert_only_if_vtcm
func.func @convert_only_if_vtcm(%x_ddr  : memref<2048xf32>,
                                %y_ddr  : memref<2048xf32>,
                                %x_vtcm : memref<2048xf32, 1>) {

// CHECK:  %[[DDR:.+]] = memref.alloc() : memref<2048xf32>
   %ddr = memref.alloc()  : memref<2048xf32>

// ON: %[[VTCM:.+]] = hexagonmem.alloc() : memref<2048xf32, 1>
    %vtcm = memref.alloc()  : memref<2048xf32, 1>

// CHECK:  memref.copy %{{.*}}, %[[DDR]] : memref<2048xf32> to memref<2048xf32>
    memref.copy %x_ddr, %ddr : memref<2048xf32> to memref<2048xf32>

// ON: hexagonmem.copy %[[DDR]], %[[VTCM]] : memref<2048xf32> to memref<2048xf32, 1>
    memref.copy %ddr, %vtcm : memref<2048xf32> to memref<2048xf32,1>

// CHECK: memref.copy %[[VTCM]], %{{.*}} : memref<2048xf32, 1> to memref<2048xf32, 1>
    memref.copy %vtcm, %x_vtcm : memref<2048xf32, 1> to memref<2048xf32, 1>

// ON: hexagonmem.copy %{{.*}}, %{{.*}} : memref<2048xf32, 1> to memref<2048xf32>
    memref.copy %x_vtcm, %y_ddr : memref<2048xf32, 1> to memref<2048xf32>

// CHECK: memref.dealloc %[[DDR]] : memref<2048xf32>
    memref.dealloc %ddr : memref<2048xf32>

// ON: hexagonmem.dealloc %[[VTCM]] : memref<2048xf32, 1>
    memref.dealloc %vtcm : memref<2048xf32, 1>
    return
}
