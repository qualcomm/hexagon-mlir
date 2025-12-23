// RUN: linalg-hexagon-opt -split-input-file %s -affine-pipeline-fusion | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @add_conv(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<5x5xf16>, %arg3: memref<252x252xf16>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xf16>
    affine.for %arg4 = 0 to 256 {
      affine.for %arg5 = 0 to 256 {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<256x256xf16>
        %1 = affine.load %arg1[%arg4, %arg5] : memref<256x256xf16>
        %2 = arith.addf %0, %1 : f16
        affine.store %2, %alloc[%arg4, %arg5] : memref<256x256xf16>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x252x252xf16>
    %c0 = arith.constant 0 : index
        affine.for %arg6 = 0 to 252 {
          affine.for %arg7 = 0 to 252 {
              affine.for %arg9 = 0 to 5 {
                affine.for %arg10 = 0 to 5 {
                  %0 = affine.apply #map(%arg6, %arg9)
                  %1 = affine.apply #map(%arg7, %arg10)
                  %2 = affine.load %alloc[%0, %1] : memref<256x256xf16>
                  %3 = affine.load %arg2[%arg9, %arg10] : memref<5x5xf16>
                  %4 = affine.load %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
                  %5 = arith.mulf %2, %3 : f16
                  %6 = arith.addf %4, %5 : f16
                  affine.store %6, %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
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
// CHECK: [[MAP0:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d0 + 4)>
// CHECK: [[MAP1:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d1 + 4)>
// CHECK: @add_conv

// Remainder of src loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.if
// Pipelined section
// CHECK: affine.for [[IV0:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: affine.for [[IV1:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: [[T0:%[a-zA-Z0-9_]+]] = affine.apply [[MAP0]]([[IV0]], [[IV1]])
// CHECK: [[T1:%[a-zA-Z0-9_]+]] = affine.apply [[MAP1]]([[IV0]], [[IV1]])
// CHECK: affine.if #{{[a-zA-Z0-9_]+}}([[IV0]], [[IV1]])
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]]{{\]}}
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]]{{\]}}
// Original dst loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 5
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 5

// -----
// CHECK-LABEL: -----

#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @add_conv_chan(
    %arg0: memref<256x256x4xf16>,
    %arg1: memref<256x256x4xf16>,
    %arg2: memref<5x5x4xf16>,
    %arg3: memref<252x252xf16>
  ) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256x4xf16>
    affine.for %arg4 = 0 to 256 {
      affine.for %arg5 = 0 to 256 {
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<256x256x4xf16>
          %1 = affine.load %arg1[%arg4, %arg5, %arg6] : memref<256x256x4xf16>
          %2 = arith.addf %0, %1 : f16
          affine.store %2, %alloc[%arg4, %arg5, %arg6] : memref<256x256x4xf16>
        }
      }
    }

    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x252x252xf16>
    %c0 = arith.constant 0 : index

    affine.for %arg7 = 0 to 252 {
      affine.for %arg8 = 0 to 252 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            affine.for %arg11 = 0 to 4 {
              %0 = affine.apply #map(%arg7, %arg9)
              %1 = affine.apply #map(%arg8, %arg10)
              %2 = affine.load %alloc[%0, %1, %arg11] : memref<256x256x4xf16>
              %3 = affine.load %arg2[%arg9, %arg10, %arg11] : memref<5x5x4xf16>
              %4 = affine.load %alloc_0[%c0, %c0, %arg7, %arg8] : memref<1x1x252x252xf16>
              %5 = arith.mulf %2, %3 : f16
              %6 = arith.addf %4, %5 : f16
              affine.store %6, %alloc_0[%c0, %c0, %arg7, %arg8] : memref<1x1x252x252xf16>
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
// CHECK: [[MAP0:#map[[:digit:]]*]] = affine_map<(d0, d1, d2) -> (d0 + 4)>
// CHECK: [[MAP1:#map[[:digit:]]*]] = affine_map<(d0, d1, d2) -> (d1 + 4)>
// CHECK: [[MAP2:#map[[:digit:]]*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: @add_conv_chan

// Remainder of src loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 4
// CHECK: affine.if
// Pipelined section
// CHECK: affine.for [[IV0:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: affine.for [[IV1:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: affine.for [[IV2:%[a-zA-Z0-9_]+]] = 0 to 4
// CHECK: [[T0:%[a-zA-Z0-9_]+]] = affine.apply [[MAP0]]([[IV0]], [[IV1]], [[IV2]])
// CHECK: [[T1:%[a-zA-Z0-9_]+]] = affine.apply [[MAP1]]([[IV0]], [[IV1]], [[IV2]])
// CHECK: [[T2:%[a-zA-Z0-9_]+]] = affine.apply [[MAP2]]([[IV0]], [[IV1]], [[IV2]])
// CHECK: affine.if #{{[a-zA-Z0-9_]+}}([[IV0]], [[IV1]])
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]], [[T2]]{{\]}}
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]], [[T2]]{{\]}}
// Original dst loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 5
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 5
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 4

// -----
// CHECK-LABEL: -----

// Multiple dst accesses

#map = affine_map<(d0) -> (d0 + 4)>
module {
  func.func @add_conv_unroll(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<5x5xf16>, %arg3: memref<252x252xf16>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xf16>
    affine.for %arg4 = 0 to 256 {
      affine.for %arg5 = 0 to 256 {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<256x256xf16>
        %1 = affine.load %arg1[%arg4, %arg5] : memref<256x256xf16>
        %2 = arith.addf %0, %1 : f16
        affine.store %2, %alloc[%arg4, %arg5] : memref<256x256xf16>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x252x252xf16>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
        affine.for %arg6 = 0 to 252 {
          affine.for %arg7 = 0 to 252 {
                  %2 = affine.load %alloc[%arg6, %arg7] : memref<256x256xf16>
                  %3 = affine.load %arg2[%c0, %c0] : memref<5x5xf16>
                  %4 = affine.load %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
                  %5 = arith.mulf %2, %3 : f16
                  %6 = arith.addf %4, %5 : f16
                  affine.store %6, %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
                  %10 = affine.apply #map(%arg6)
                  %11 = affine.apply #map(%arg7)
                  %12 = affine.load %alloc[%10, %11] : memref<256x256xf16>
                  %13 = affine.load %arg2[%c4, %c4] : memref<5x5xf16>
                  %14 = affine.load %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
                  %15 = arith.mulf %12, %13 : f16
                  %16 = arith.addf %14, %15 : f16
                  affine.store %16, %alloc_0[%c0, %c0, %arg6, %arg7] : memref<1x1x252x252xf16>
          }
        }
    %collapse_shape = memref.collapse_shape %alloc_0 [[0, 1, 2], [3]] : memref<1x1x252x252xf16> into memref<252x252xf16>
    memref.copy %collapse_shape, %arg3 : memref<252x252xf16> to memref<252x252xf16>
    return
  }
}

// Pipeline map
// CHECK: [[MAP0:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d0 + 4)>
// CHECK: [[MAP1:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d1 + 4)>
// CHECK: @add_conv_unroll

// Remainder of src loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.if
// Pipelined section
// CHECK: affine.for [[IV0:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: affine.for [[IV1:%[a-zA-Z0-9_]+]] = 0 to 252
// CHECK: [[T0:%[a-zA-Z0-9_]+]] = affine.apply [[MAP0]]([[IV0]], [[IV1]])
// CHECK: [[T1:%[a-zA-Z0-9_]+]] = affine.apply [[MAP1]]([[IV0]], [[IV1]])
// CHECK: affine.if #{{[a-zA-Z0-9_]+}}([[IV0]], [[IV1]])
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]]{{\]}}
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]]{{\]}}
// Original dst loop
// CHECK: affine.load
