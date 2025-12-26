// RUN: linalg-hexagon-opt %s -affine-tiling="cache-size=12" | FileCheck %s

func.func @test_tile_size(%A: memref<768x768xf32>, %B: memref<768x768xf32>, %C: memref<768x768xf32>, %D: memref<768x768xf32>) {
  affine.for %i = 0 to 768 {
    affine.for %j = 0 to 768 {
      %x = affine.load %A[%i, %j] : memref<768x768xf32>
      %y = affine.load %B[%i, %j] : memref<768x768xf32>
      %z = affine.load %C[%i, %j] : memref<768x768xf32>
    }
  }
  return
}

// CHECK-LABEL: @test_tile_size
// inner loop footprint: 12 bytes
// cache size: 12 KiB = 2^12 * 3
// --> can fit 1024 = 32 * 32 iterations

// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 768 step 32
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 768 step 32

func.func @test_tile_size_bigger(%A: memref<768x768xf32>, %B: memref<768x768xf32>, %C: memref<768x768xf32>, %D: memref<768x768xf32>) {
  affine.for %i = 0 to 768 {
    affine.for %j = 0 to 768 {
      %x = affine.load %A[%i, %j] : memref<768x768xf32>
      %y = affine.load %B[%i, %j] : memref<768x768xf32>
      %z = affine.load %C[%i, %j] : memref<768x768xf32>
      %w = affine.load %D[%i, %j] : memref<768x768xf32>
    }
  }
  return
}

// CHECK-LABEL: @test_tile_size_bigger
// inner loop footprint: 16 bytes
// --> can fit 2^8 * 3 = 768 iterations
//     = 27.71^2
// round down: 24
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 768 step 24
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 768 step 24

func.func @test_tile_size_bigger_pot(%A: memref<768x768xf32>, %B: memref<768x768xf32>, %C: memref<768x768xf32>, %D: memref<768x768xf32>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %x = affine.load %A[%i, %j] : memref<768x768xf32>
      %y = affine.load %B[%i, %j] : memref<768x768xf32>
      %z = affine.load %C[%i, %j] : memref<768x768xf32>
      %w = affine.load %D[%i, %j] : memref<768x768xf32>
    }
  }
  return
}

// CHECK-LABEL: @test_tile_size_bigger_pot
// round down: 16
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256 step 16
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 256 step 16

func.func @test_tile_fits(%A: memref<768x768xf32>, %B: memref<768x768xf32>, %C: memref<768x768xf32>, %D: memref<768x768xf32>) {
  affine.for %i = 0 to 25 {
    affine.for %j = 0 to 25 {
      %x = affine.load %A[%i, %j] : memref<768x768xf32>
      %y = affine.load %B[%i, %j] : memref<768x768xf32>
      %z = affine.load %C[%i, %j] : memref<768x768xf32>
    }
  }
  return
}

// CHECK-LABEL: @test_tile_fits
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 25 {
// CHECK: affine.for %{{[a-zA-Z0-9_]+}} = 0 to 25 {
// CHECK-NOT: affine.for
// CHECK: }
// CHECK: }
