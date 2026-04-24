// Verify that the vtcm-budget option controls tile sizes.
//
// Default (2 MB) budget: tiles to 64×4096 / 4096×64 / 64×64.
// Reduced (256 KB) budget: tiles to  8×1024 / 1024×64 /  8×64.
//
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(vtcm-tiling), \
// RUN:     canonicalize, one-shot-bufferize, \
// RUN:     func.func(buffer-loop-hoisting), canonicalize, \
// RUN:     buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s --check-prefix=DEFAULT
//
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(vtcm-tiling{vtcm-budget=262144}), \
// RUN:     canonicalize, one-shot-bufferize, \
// RUN:     func.func(buffer-loop-hoisting), canonicalize, \
// RUN:     buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s --check-prefix=SMALL

#map  = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

// A matmul large enough that tiling is required under both budgets,
// but the tile sizes must differ.
//   X: 2048×8192 f16   Y: 8192×1024 f16   Z: 2048×1024 f32
func.func @matmul_budget(%x: memref<2048x8192xf16>,
                         %y: memref<8192x1024xf16>,
                         %z: memref<2048x1024xf32>) {
  %t0 = bufferization.to_tensor %x restrict writable : memref<2048x8192xf16> to tensor<2048x8192xf16>
  %t1 = bufferization.to_tensor %y restrict writable : memref<8192x1024xf16> to tensor<8192x1024xf16>
  %t2 = bufferization.to_tensor %z restrict writable : memref<2048x1024xf32> to tensor<2048x1024xf32>
  %r = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%t0, %t1 : tensor<2048x8192xf16>, tensor<8192x1024xf16>)
      outs(%t2 : tensor<2048x1024xf32>) {
  ^bb0(%a: f16, %b: f16, %c: f32):
    %0 = arith.extf %a : f16 to f32
    %1 = arith.extf %b : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %c, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<2048x1024xf32>
  bufferization.materialize_in_destination %r in writable %z
      : (tensor<2048x1024xf32>, memref<2048x1024xf32>) -> ()
  return
}

// ---------------------------------------------------------------
// DEFAULT (2 MB): tiles 64×4096 / 4096×64 / 64×64
// ---------------------------------------------------------------
// VTCM tile allocations (hoisted before loops)
// DEFAULT-DAG: memref.alloc() {{.*}} : memref<64x4096xf16, 1>
// DEFAULT-DAG: memref.alloc() {{.*}} : memref<4096x64xf16, 1>
// DEFAULT-DAG: memref.alloc() {{.*}} : memref<64x64xf32, 1>
//
// Outer loops: I step 64, J step 4096, K step 64
// DEFAULT: scf.for {{.*}} step %c64
// DEFAULT:   scf.for {{.*}} step %c4096
// DEFAULT:     scf.for {{.*}} step %c64

// ---------------------------------------------------------------
// SMALL (256 KB): tiles 8×1024 / 1024×64 / 8×64
// ---------------------------------------------------------------
// VTCM tile allocations (hoisted before loops)
// SMALL-DAG: memref.alloc() {{.*}} : memref<8x1024xf16, 1>
// SMALL-DAG: memref.alloc() {{.*}} : memref<1024x64xf16, 1>
// SMALL-DAG: memref.alloc() {{.*}} : memref<8x64xf32, 1>
//
// Outer loops: I step 8, J step 1024, K step 64
// SMALL: scf.for {{.*}} step %c8
// SMALL:   scf.for {{.*}} step %c1024
// SMALL:     scf.for {{.*}} step %c64
