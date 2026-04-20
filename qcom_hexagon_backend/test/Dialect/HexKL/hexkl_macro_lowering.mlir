// RUN: linalg-hexagon-opt %s --pass-pipeline="builtin.module(func.func(lower-hexkl-matmul-to-macro))" -split-input-file | FileCheck %s

// ============================================================================
// Test 1: Basic single matmul lowering with all transformations
// ============================================================================

module @single_matmul {
  // CHECK-LABEL: func.func @test_single_matmul
  func.func @test_single_matmul(%A : memref<32x64xf16>, %B : memref<64x128xf16>, %C : memref<32x128xf16>) {

    // CHECK: hexkl.macro_initialize
    // CHECK: hexkl.macro_lock_hmx
    // CHECK: hexkl.macro_rm_to_ah_f16_inplace
    // CHECK: hexkl.qurt_mem_cache_clean
    // CHECK: hexkl.qurt_mem_cache_clean
    // CHECK: hexkl.qurt_mem_cache_clean
    // CHECK: hexkl.macro_mm_f16
    // CHECK: hexkl.macro_ah_to_rm_f16_inplace
    // CHECK: hexkl.macro_unlock_hmx
    // CHECK: hexkl.macro_finalize

    // CHECK-NOT: hexkl.matmul
    hexkl.matmul ins(%A, %B : memref<32x64xf16>, memref<64x128xf16>) outs(%C : memref<32x128xf16>)

    return
  }
}

// -----

// ============================================================================
// Test 2: Consecutive matmuls share init/lock/unlock/finalize
// ============================================================================

module @consecutive_matmuls {
  // CHECK-LABEL: func.func @test_consecutive_matmuls
  func.func @test_consecutive_matmuls(%A : memref<32x64xf16>, %B : memref<64x128xf16>,
                                       %C : memref<32x128xf16>, %D : memref<128x256xf16>,
                                       %E : memref<32x256xf16>) {

    // CHECK: hexkl.macro_initialize
    // CHECK: hexkl.macro_lock_hmx

    // First matmul
    // CHECK: hexkl.macro_mm_f16

    // CHECK-NOT: hexkl.macro_unlock_hmx
    // CHECK-NOT: hexkl.macro_finalize

    // Second matmul (no new init/lock)
    // CHECK: hexkl.macro_mm_f16

    // CHECK: hexkl.macro_unlock_hmx
    // CHECK: hexkl.macro_finalize

    // CHECK-NOT: hexkl.matmul
    hexkl.matmul ins(%A, %B : memref<32x64xf16>, memref<64x128xf16>) outs(%C : memref<32x128xf16>)
    hexkl.matmul ins(%C, %D : memref<32x128xf16>, memref<128x256xf16>) outs(%E : memref<32x256xf16>)

    return
  }
}

// -----

// ============================================================================
// Test 3: Matmul inside loop - init/lock before loop, unlock/finalize after
// ============================================================================

module @loop_matmul {
  // CHECK-LABEL: func.func @test_loop_matmul
  func.func @test_loop_matmul(%A : memref<32x64xf16>, %B : memref<64x128xf16>, %C : memref<32x128xf16>) {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    // CHECK: hexkl.macro_initialize
    // CHECK: hexkl.macro_lock_hmx

    // CHECK: scf.for
    scf.for %i = %c0 to %c10 step %c1 {
      // CHECK: hexkl.macro_mm_f16
      // CHECK-NOT: hexkl.matmul
      hexkl.matmul ins(%A, %B : memref<32x64xf16>, memref<64x128xf16>) outs(%C : memref<32x128xf16>)
      scf.yield
    }

    // CHECK: hexkl.macro_unlock_hmx
    // CHECK: hexkl.macro_finalize

    return
  }
}

// -----

// ============================================================================
// Test 4: Nested loops - init/lock at outermost loop
// ============================================================================

module @nested_loops {
  // CHECK-LABEL: func.func @test_nested_loops
  func.func @test_nested_loops(%A : memref<32x64xf16>, %B : memref<64x128xf16>, %C : memref<32x128xf16>) {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index

    // CHECK: hexkl.macro_initialize
    // CHECK: hexkl.macro_lock_hmx

    // CHECK: scf.for
    scf.for %i = %c0 to %c5 step %c1 {
      // CHECK: scf.for
      scf.for %j = %c0 to %c10 step %c1 {
        // CHECK: hexkl.macro_mm_f16
        // CHECK-NOT: hexkl.matmul
        hexkl.matmul ins(%A, %B : memref<32x64xf16>, memref<64x128xf16>) outs(%C : memref<32x128xf16>)
        scf.yield
      }
      scf.yield
    }

    // CHECK: hexkl.macro_unlock_hmx
    // CHECK: hexkl.macro_finalize

    return
  }
}
