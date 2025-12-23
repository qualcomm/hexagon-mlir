// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul))' | FileCheck %s

// Test decompose-hexkl-matmul pass with both static and dynamic shapes

// Test 1: Static shapes (baseline case)

func.func @test_static_shapes(%arg0: memref<32x64xf16>,
                               %arg1: memref<64x128xf16>,
                               %output: memref<32x128xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<32x64xf16>, memref<64x128xf16>)
                 outs(%output : memref<32x128xf32>)
    return
}

// CHECK-LABEL: func.func @test_static_shapes
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:         hexkl.micro_hmx_rm_to_ah_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_acc_clear_f16
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_rm_to_wh_f16
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:         hexkl.micro_hmx_acc_read_f16
// CHECK:         hexkl.micro_hmx_ah_to_rm_f16
// CHECK:         hexkl.micro_hmx_copy_f16_to_f32_submatrix
// CHECK:       hexagonmem.dealloc

// -----

// Test 2: Fully dynamic shapes

func.func @test_fully_dynamic(%arg0: memref<?x?xf16>,
                               %arg1: memref<?x?xf16>,
                               %output: memref<?x?xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
                 outs(%output : memref<?x?xf32>)
    return
}

// CHECK-LABEL: func.func @test_fully_dynamic
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:         hexkl.micro_hmx_rm_to_ah_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_acc_clear_f16
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_rm_to_wh_f16
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:         hexkl.micro_hmx_acc_read_f16
// CHECK:         hexkl.micro_hmx_ah_to_rm_f16
// CHECK:         hexkl.micro_hmx_copy_f16_to_f32_submatrix
// CHECK:       hexagonmem.dealloc

// -----

// Test 3: Partially dynamic (K dimension dynamic)

func.func @test_partial_dynamic_k(%arg0: memref<32x?xf16>,
                                   %arg1: memref<?x128xf16>,
                                   %output: memref<32x128xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<32x?xf16>, memref<?x128xf16>)
                 outs(%output : memref<32x128xf32>)
    return
}

// CHECK-LABEL: func.func @test_partial_dynamic_k
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:         hexkl.micro_hmx_rm_to_ah_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_acc_clear_f16
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_rm_to_wh_f16
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:         hexkl.micro_hmx_acc_read_f16
// CHECK:         hexkl.micro_hmx_ah_to_rm_f16
// CHECK:         hexkl.micro_hmx_copy_f16_to_f32_submatrix
// CHECK:       hexagonmem.dealloc

// -----

// Test 4: Mixed dynamic (M and N dynamic, K static)

func.func @test_mixed_dynamic(%arg0: memref<?x64xf16>,
                               %arg1: memref<64x?xf16>,
                               %output: memref<?x?xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<?x64xf16>, memref<64x?xf16>)
                 outs(%output : memref<?x?xf32>)
    return
}

// CHECK-LABEL: func.func @test_mixed_dynamic
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:         hexkl.micro_hmx_rm_to_ah_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_acc_clear_f16
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_rm_to_wh_f16
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:         hexkl.micro_hmx_acc_read_f16
// CHECK:         hexkl.micro_hmx_ah_to_rm_f16
// CHECK:         hexkl.micro_hmx_copy_f16_to_f32_submatrix
// CHECK:       hexagonmem.dealloc

// -----

// Test 5: Small static matrix (single tile)

func.func @test_small_static(%arg0: memref<32x32xf16>,
                              %arg1: memref<32x32xf16>,
                              %output: memref<32x32xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>)
                 outs(%output : memref<32x32xf32>)
    return
}

// CHECK-LABEL: func.func @test_small_static
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:       hexagonmem.dealloc

// -----

// Test 6: Large static matrix (multiple tiles)

func.func @test_large_static(%arg0: memref<64x128xf16>,
                              %arg1: memref<128x256xf16>,
                              %output: memref<64x256xf32>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<64x128xf16>, memref<128x256xf16>)
                 outs(%output : memref<64x256xf32>)
    return
}

// CHECK-LABEL: func.func @test_large_static
// CHECK:       hexagonmem.alloc
// CHECK:       hexkl.micro_hmx_setup_acc_read_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:         hexkl.micro_hmx_rm_to_ah_f16
// CHECK:       scf.for
// CHECK:         hexkl.micro_hmx_acc_clear_f16
// CHECK:         scf.for
// CHECK:           hexkl.micro_hmx_rm_to_wh_f16
// CHECK:           hexkl.micro_hmx_mm_f16
// CHECK:         hexkl.micro_hmx_acc_read_f16
// CHECK:         hexkl.micro_hmx_ah_to_rm_f16
// CHECK:         hexkl.micro_hmx_copy_f16_to_f32_submatrix
// CHECK:       hexagonmem.dealloc
