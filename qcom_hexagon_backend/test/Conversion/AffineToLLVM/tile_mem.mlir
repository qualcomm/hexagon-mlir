// RUN: linalg-hexagon-opt -affine-tile-memory='cache-size=16' %s | FileCheck %s

#id = affine_map<(d0) -> (d0)>
#plus_16 = affine_map<(d0) -> (d0 + 16)>
#plus_32 = affine_map<(d0) -> (d0 + 32)>
#plus_64 = affine_map<(d0) -> (d0 + 64)>
#plus_64_128 = affine_map<(d0) -> (d0 + 64, 128)>
#minus_16_0 = affine_map<(d0) -> (d0 - 16, 0)>
#double = affine_map<(d0) -> (2 * d0)>
#double_plus_16 = affine_map<(d0) -> (2 * d0 + 16)>

// CHECK-LABEL: @no_tile_if_no_step_loops
func.func @no_tile_if_no_step_loops(%arg0: memref<64x64xf32>) -> () {
  // Pass processes only loops with step > 1.
  // CHECK-NOT: memref.alloc()
  // CHECK: affine.for
  affine.for %i0 = 0 to 64 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 64 {
      // CHECK-NOT: memref.alloc()
      %0 = affine.load %arg0[%i0, %j0] : memref<64x64xf32>
      affine.store %0, %arg0[%i0, %j0] : memref<64x64xf32>
    }
  }
  return
}

// CHECK-LABEL: @temp_buffer_fixed_size_subview
func.func @temp_buffer_fixed_size_subview(%arg0: memref<128x128xf32>) -> () {
  // Tile size (64x64) * element size (4) = 16384 bytes, which fits in cache.
  // Accesses are always within bounds of the subview.
  // CHECK: %[[ALLOC:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<64x64xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 128 step 64 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 128 step 64 {
      // CHECK-NOT: memref.alloc
      // CHECK-NOT: index.mins
      // CHECK-NOT: index.maxs
      // CHECK: memref.subview
      // CHECK: memref.copy %{{[a-zA-Z0-9_-]+}}, %[[ALLOC]]
      // CHECK: affine.for
      affine.for %i1 = #id(%i0) to #plus_64(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #id(%j0) to #plus_64(%j0) {
          // CHECK: affine.load %[[ALLOC]]
          %0 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[ALLOC]]
          affine.store %0, %arg0[%i1, %j1] : memref<128x128xf32>
        }
      }
      // CHECK: memref.copy %[[ALLOC]], %{{[a-zA-Z0-9_-]+}}
    }
  }
  // CHECK: memref.dealloc %[[ALLOC]]
  return
}

// CHECK-LABEL: @temp_buffer_lower_bound_out_of_bounds_subview
func.func @temp_buffer_lower_bound_out_of_bounds_subview(%arg0: memref<128x128xf32>) -> () {
  // CHECK: %[[ALLOC:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<48x48xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 128 step 64 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 128 step 64 {
      // CHECK: affine.apply
      // CHECK: index.maxs %c0{{[_0-9]*}}, %{{[a-zA-Z0-9_-]+}}
      // CHECK: index.sub %c48{{[_0-9]*}}, %{{[a-zA-Z0-9_-]+}}
      // CHECK: affine.apply
      // CHECK: index.maxs %c0{{[_0-9]*}}, %{{[a-zA-Z0-9_-]+}}
      // CHECK: index.sub %c48{{[_0-9]*}}, %{{[a-zA-Z0-9_-]+}}
      // CHECK-DAG: %[[FULL_SUBVIEW:[a-zA-Z0-9_-]+]] = memref.subview %{{[a-zA-Z0-9]+}}
      // CHECK-DAG: %[[SUBVIEW:[a-zA-Z0-9_-]+]] = memref.subview %[[ALLOC]]
      // CHECK: memref.copy %[[FULL_SUBVIEW]], %[[SUBVIEW]]
      // CHECK: affine.for
      affine.for %i1 = max #minus_16_0(%i0) to #plus_32(%i0) {
        // CHECK: affine.for
        affine.for %j1 = max #minus_16_0(%j0) to #plus_32(%j0) {
          // CHECK: affine.load %[[ALLOC]]
          %0 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          affine.store %0, %arg0[%i1, %j1] : memref<128x128xf32>
        }
      }
      // CHECK: memref.copy %[[SUBVIEW]], %[[FULL_SUBVIEW]]
    }
  }
  return
}

// CHECK-LABEL: @temp_buffer_upper_bound_out_of_bounds_subview
func.func @temp_buffer_upper_bound_out_of_bounds_subview(%arg0: memref<128x128xf32>) -> () {
  // CHECK: %[[ALLOC:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<32x32xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 128 step 48 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 128 step 48 {
      // CHECK: affine.apply
      // CHECK: memref.dim
      // CHECK: index.sub
      // CHECK: index.mins
      // CHECK: affine.apply
      // CHECK: memref.dim
      // CHECK: index.sub
      // CHECK: index.mins
      // CHECK-DAG: %[[FULL_SUBVIEW:[a-zA-Z0-9_-]+]] = memref.subview %{{[a-zA-Z0-9]+}}
      // CHECK-DAG: %[[SUBVIEW:[a-zA-Z0-9_-]+]] = memref.subview %[[ALLOC]]
      // CHECK: memref.copy %[[FULL_SUBVIEW]], %[[SUBVIEW]]
      // CHECK: affine.for
      affine.for %i1 = #plus_32(%i0) to min #plus_64_128(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #plus_32(%j0) to min #plus_64_128(%j0) {
          // CHECK: affine.load %[[ALLOC]]
          %0 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          affine.store %0, %arg0[%i1, %j1] : memref<128x128xf32>
        }
      }
      // CHECK: memref.copy %[[SUBVIEW]], %[[FULL_SUBVIEW]]
    }
  }
  return
}

// CHECK-LABEL: @cache_allocation_priority_reuse
func.func @cache_allocation_priority_reuse() -> () {
  // Both tiles have size 16KiB, so only one fits
  // CHECK: %[[ORIG_ARG0:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<128x128xf32>
  %arg0 = memref.alloc() : memref<128x128xf32>
  // CHECK: %[[ORIG_ARG1:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<128x128xf32>
  %arg1 = memref.alloc() : memref<128x128xf32>
  // This loop: arg1 has greater reuse, should be tiled
  // CHECK: %[[CACHED_ARG1:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<64x64xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 128 step 64 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 128 step 64 {
      // arg0 should be tiled, so copy inside the tiled loop
      // CHECK: memref.copy
      // CHECK: affine.for
      affine.for %i1 = #id(%i0) to #plus_64(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #id(%j0) to #plus_64(%j0) {
          // CHECK: affine.load %[[ORIG_ARG0]]
          %0 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.load %[[CACHED_ARG1]]
          %1 = affine.load %arg1[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.load %[[CACHED_ARG1]]
          %2 = affine.load %arg1[%i1, %j1] : memref<128x128xf32>
          %3 = arith.addf %1, %2 : f32
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[ORIG_ARG0]]
          affine.store %0, %arg0[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[CACHED_ARG1]]
          affine.store %3, %arg1[%i1, %j1] : memref<128x128xf32>
        }
      }
    }
  }
  // CHECK: memref.dealloc %[[CACHED_ARG1]]
  // This loop: arg0 has greater reuse, should be tiled
  // CHECK: %[[CACHED_ARG0:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<64x64xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 128 step 64 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 128 step 64 {
      // CHECK: memref.copy
      // CHECK: affine.for
      affine.for %i1 = #id(%i0) to #plus_64(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #id(%j0) to #plus_64(%j0) {
          // CHECK: affine.load %[[CACHED_ARG0]]
          %0 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.load %[[CACHED_ARG0]]
          %1 = affine.load %arg0[%i1, %j1] : memref<128x128xf32>
          %2 = arith.addf %0, %1 : f32
          // CHECK: affine.load %[[ORIG_ARG1]]
          %3 = affine.load %arg1[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[CACHED_ARG0]]
          affine.store %2, %arg0[%i1, %j1] : memref<128x128xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[ORIG_ARG1]]
          affine.store %3, %arg1[%i1, %j1] : memref<128x128xf32>
        }
      }
    }
  }
  // CHECK: memref.dealloc %[[CACHED_ARG0]]
  return
}

// CHECK-LABEL: @cache_allocation_priority_promotion
func.func @cache_allocation_priority_promotion() -> () {
  // CHECK: %[[ORIG_ARG0:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<48x48xf32>
  %arg0 = memref.alloc() : memref<48x48xf32>
  // CHECK: %[[ORIG_ARG1:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<48x48xf32>
  %arg1 = memref.alloc() : memref<48x48xf32>
  // 16x16 tiles, 48x48 memrefs, so one fits fully + one tile
  // CHECK-DAG: %[[FULL_ARG1:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<48x48xf32>
  // CHECK-DAG: memref.copy %[[ORIG_ARG1]], %[[FULL_ARG1]]
  // CHECK-DAG: %[[TILE_ARG0:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<16x16xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 24 step 8 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 24 step 8 {
      // Access arg0 as 16x16 tiles, overlapping by 8 each iteration
      // Here we use the same reuse pattern for both memrefs,
      // to make sure promotion is independent of reuse.
      // CHECK: affine.for
      affine.for %i1 = #id(%i0) to #plus_16(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #id(%j0) to #plus_16(%j0) {
          // CHECK: affine.load %[[TILE_ARG0]]
          %0 = affine.load %arg0[%i1, %j1] : memref<48x48xf32>
          // CHECK: affine.load %[[TILE_ARG0]]
          %1 = affine.load %arg0[%i1, %j1] : memref<48x48xf32>
          %2 = arith.addf %0, %1 : f32
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[TILE_ARG0]]
          affine.store %2, %arg0[%i1, %j1] : memref<48x48xf32>
        }
      }
      // Access arg1 as 16x16 tiles with no overlap
      // Since we save more memory usage by promoting arg1,
      // arg1 should be promoted to full.
      // CHECK: affine.for
      affine.for %i1 = #double(%i0) to #double_plus_16(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #double(%j0) to #double_plus_16(%j0) {
          // CHECK: affine.load %[[FULL_ARG1]]
          %0 = affine.load %arg1[%i1, %j1] : memref<48x48xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[FULL_ARG1]]
          affine.store %0, %arg1[%i1, %j1] : memref<48x48xf32>
        }
      }
    }
  }
  // CHECK-DAG: memref.dealloc %[[FULL_ARG1]]
  // CHECK-DAG: memref.dealloc %[[TILE_ARG0]]
  // Marker statement
  // CHECK: arith.constant 1234
  %c1234 = arith.constant 1234 : i32
  // CHECK-DAG: %[[FULL_ARG0:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<48x48xf32>
  // CHECK-DAG: memref.copy %[[ORIG_ARG0]], %[[FULL_ARG0]]
  // CHECK-DAG: %[[TILE_ARG1:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<16x16xf32>
  // CHECK: affine.for
  affine.for %i0 = 0 to 24 step 8 {
    // CHECK: affine.for
    affine.for %j0 = 0 to 24 step 8 {
      // Access arg1 as 16x16 tiles, overlapping by 8 each iteration
      // CHECK: affine.for
      affine.for %i1 = #id(%i0) to #plus_16(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #id(%j0) to #plus_16(%j0) {
          // CHECK: affine.load %[[TILE_ARG1]]
          %0 = affine.load %arg1[%i1, %j1] : memref<48x48xf32>
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[TILE_ARG1]]
          affine.store %0, %arg1[%i1, %j1] : memref<48x48xf32>
        }
      }
      // Access arg0 as 16x16 tiles with no overlap.
      // arg0 should be promoted here.
      // CHECK: affine.for
      affine.for %i1 = #double(%i0) to #double_plus_16(%i0) {
        // CHECK: affine.for
        affine.for %j1 = #double(%j0) to #double_plus_16(%j0) {
          // CHECK: affine.load %[[FULL_ARG0]]
          %0 = affine.load %arg0[%i1, %j1] : memref<48x48xf32>
          // CHECK: affine.load %[[FULL_ARG0]]
          %1 = affine.load %arg0[%i1, %j1] : memref<48x48xf32>
          %2 = arith.addf %0, %1 : f32
          // CHECK: affine.store %{{[a-zA-Z0-9_-]+}}, %[[FULL_ARG0]]
          affine.store %2, %arg0[%i1, %j1] : memref<48x48xf32>
        }
      }
    }
  }
  // CHECK-DAG: memref.dealloc %[[FULL_ARG0]]
  // CHECK-DAG: memref.dealloc %[[TILE_ARG1]]
  return
}

// CHECK-LABEL: @no_copy_back_on_read_only
func.func @no_copy_back_on_read_only(%arg0: memref<64x64xf32>) -> () {
  // CHECK: %[[ALLOC:[a-zA-Z0-9_-]+]] = memref.alloc() : memref<64x64xf32>
  // CHECK: memref.copy %{{[a-zA-Z0-9_-]+}}, %[[ALLOC]]
  affine.for %i0 = 0 to 64 step 32 {
    affine.for %j0 = 0 to 64 step 32 {
      affine.for %i1 = #id(%i0) to #plus_32(%i0) {
        affine.for %j1 = #id(%j0) to #plus_32(%j0) {
          %0 = affine.load %arg0[%i1, %j1] : memref<64x64xf32>
        }
      }
    }
  }
  // CHECK-NOT: memref.copy %[[ALLOC]]
  // CHECK: memref.dealloc %[[ALLOC]]
  return
}
