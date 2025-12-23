// RUN: linalg-hexagon-opt -split-input-file %s -affine-pipeline-fusion | FileCheck %s

module {
  func.func @skewed(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
    %cst_2 = arith.constant 2.0 : f32
    %cst_5 = arith.constant 5.0 : f32

    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %a = affine.load %A[%i, %j] : memref<256x256xf32>
        %mul = arith.mulf %a, %cst_2 : f32
        affine.store %mul, %C[%i, %j] : memref<256x256xf32>
      }
    }

    affine.for %i = 0 to 128 {
      affine.for %j = 0 to 128 {
        affine.for %k = 0 to affine_map<(d0) -> (d0)>(%j) {
          %1 = affine.load %C[%i + %k, %j] : memref<256x256xf32>
          %add = arith.addf %1, %1 : f32
          affine.store %add, %B[%i, %j] : memref<256x256xf32>
        }
      }
    }

    return
  }
}

// Pipeline map
// CHECK: [[MAP0:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d0 + d1 - 1)>
// CHECK: [[MAP1:#map[[:digit:]]*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: @skewed

// Remainder: C escapes, so entirely preserved, minus pipelined
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256
// CHECK: affine.if
// 3 disjuncts; don't check all of them
// Pipelined section
// CHECK: affine.for [[IV0:%[a-zA-Z0-9_]+]] = 0 to 128
// CHECK: affine.for [[IV1:%[a-zA-Z0-9_]+]] = 0 to 128
// CHECK: [[T0:%[a-zA-Z0-9_]+]] = affine.apply [[MAP0]]([[IV0]], [[IV1]])
// CHECK: [[T1:%[a-zA-Z0-9_]+]] = affine.apply [[MAP1]]([[IV0]], [[IV1]])
// CHECK: %{{[a-zA-Z0-9_]+}} = affine.load %{{[a-zA-Z0-9_]+}}{{\[}}[[T0]], [[T1]]{{\]}}
// Original dst loop
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to #{{[a-zA-Z0-9_]+}}([[IV1]])
