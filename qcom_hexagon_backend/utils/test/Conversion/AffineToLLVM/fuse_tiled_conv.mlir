// RUN: linalg-hexagon-opt -split-input-file %s -affine-pipeline-fusion | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 16, 252)>
#map3 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @add_conv_tiled(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<5x5xf16>, %arg3: memref<252x252xf16>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xf16>
    affine.for %arg4 = 0 to 256 step 16 {
      affine.for %arg5 = 0 to 256 step 16 {
        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
          affine.for %arg7 = #map(%arg5) to #map1(%arg5) {
            %0 = affine.load %arg0[%arg6, %arg7] : memref<256x256xf16>
            %1 = affine.load %arg1[%arg6, %arg7] : memref<256x256xf16>
            %2 = arith.addf %0, %1 : f16
            affine.store %2, %alloc[%arg6, %arg7] : memref<256x256xf16>
          }
        }
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x252x252xf16>
    %c0 = arith.constant 0 : index
    affine.for %arg4 = 0 to 252 step 16 {
      affine.for %arg5 = 0 to 252 step 16 {
        affine.for %arg6 = #map(%arg4) to min #map2(%arg4) {
          affine.for %arg7 = #map(%arg5) to min #map2(%arg5) {
            affine.for %arg8 = 0 to 5 {
              affine.for %arg9 = 0 to 5 {
                %0 = affine.apply #map3(%arg6, %arg8)
                %1 = affine.apply #map3(%arg7, %arg9)
                %2 = affine.load %alloc[%0, %1] : memref<256x256xf16>
                %3 = affine.load %arg2[%arg8, %arg9] : memref<5x5xf16>
                %4 = affine.load %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
                %5 = arith.mulf %2, %3 : f16
                %6 = arith.addf %4, %5 : f16
                affine.store %6, %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
              }
            }
          }
        }
      }
    }
    %collapse_shape = memref.collapse_shape %alloc_0 [[0, 1, 2], [3]] : memref<1x1x252x252xf16> into memref<252x252xf16>
    memref.copy %collapse_shape, %arg3 : memref<252x252xf16> to memref<252x252xf16>
    return
  }
}

// Pipeline map
// CHECK: [[MAP0:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d0 + 16)>
// CHECK: [[MAP1:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d1 + 16)>
// CHECK: @add_conv_tiled

// Remainder of src loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256 step 16
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256 step 16
// CHECK: affine.if
// Pipelined section
// CHECK: affine.for [[IV0:%[a-zA-Z0-9_]+]] = 0 to 252 step 16
// CHECK: affine.for [[IV1:%[a-zA-Z0-9_]+]] = 0 to 252 step 16
// CHECK: [[T0:%[a-zA-Z0-9_]+]] = affine.apply [[MAP0]]([[IV0]], [[IV1]])
// CHECK: [[T1:%[a-zA-Z0-9_]+]] = affine.apply [[MAP1]]([[IV0]], [[IV1]])
// CHECK: affine.if #{{[a-zA-Z0-9_]+}}([[IV0]], [[IV1]])
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = {{#map[[:digit:]]*}}([[T0]])
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = {{#map[[:digit:]]*}}([[T1]])
// Original dst loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = {{#map[[:digit:]]*}}([[IV0]])
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = {{#map[[:digit:]]*}}([[IV1]])
