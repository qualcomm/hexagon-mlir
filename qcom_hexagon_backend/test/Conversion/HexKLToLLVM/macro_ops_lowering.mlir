// RUN: linalg-hexagon-opt %s -hexkl-to-llvm | FileCheck %s

// This test checks that each HexKL macro op lowers to an LLVM call with the
// expected extern C signature.

// macro_initialize: () -> i32
func.func @macro_initialize() {
  // CHECK-LABEL: func.func @macro_initialize
  // CHECK: llvm.call @hexkl_macro_initialize() : () -> i32
  hexkl.macro_initialize
  return
}

// macro_finalize: () -> i32
func.func @macro_finalize() {
  // CHECK-LABEL: func.func @macro_finalize
  // CHECK: llvm.call @hexkl_macro_finalize() : () -> i32
  hexkl.macro_finalize
  return
}

// macro_lock_hmx: () -> i32
func.func @macro_lock_hmx() {
  // CHECK-LABEL: func.func @macro_lock_hmx
  // CHECK: llvm.call @hexkl_macro_lock_hmx() : () -> i32
  hexkl.macro_lock_hmx
  return
}

// macro_unlock_hmx: () -> i32
func.func @macro_unlock_hmx() {
  // CHECK-LABEL: func.func @macro_unlock_hmx
  // CHECK: llvm.call @hexkl_macro_unlock_hmx() : () -> i32
  hexkl.macro_unlock_hmx
  return
}

// macro_rm_to_ah_f16_inplace: (i32, i32, !llvm.ptr) -> i32
func.func @macro_rm_to_ah(%buf: memref<32x64xf16>) {
  %m = arith.constant 32 : i32
  %k = arith.constant 64 : i32
  %ptr_idx = memref.extract_aligned_pointer_as_index %buf : memref<32x64xf16> -> index
  %ptr_i64 = arith.index_cast %ptr_idx : index to i64
  %ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr
  // CHECK-LABEL: func.func @macro_rm_to_ah
  // CHECK: llvm.call @hexkl_macro_rm_to_ah_f16_inplace({{.*}}) : (i32, i32, !llvm.ptr) -> i32
  hexkl.macro_rm_to_ah_f16_inplace(%m, %k, %ptr) : i32, i32, !llvm.ptr
  return
}

// macro_ah_to_rm_f16_inplace: (i32, i32, !llvm.ptr) -> i32
func.func @macro_ah_to_rm(%buf: memref<32x128xf16>) {
  %m = arith.constant 32 : i32
  %n = arith.constant 128 : i32
  %ptr_idx = memref.extract_aligned_pointer_as_index %buf : memref<32x128xf16> -> index
  %ptr_i64 = arith.index_cast %ptr_idx : index to i64
  %ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr
  // CHECK-LABEL: func.func @macro_ah_to_rm
  // CHECK: llvm.call @hexkl_macro_ah_to_rm_f16_inplace({{.*}}) : (i32, i32, !llvm.ptr) -> i32
  hexkl.macro_ah_to_rm_f16_inplace(%m, %n, %ptr) : i32, i32, !llvm.ptr
  return
}

// macro_mm_f16: (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
func.func @macro_mm_f16(%out: memref<32x128xf16>, %act: memref<32x64xf16>, %wgt: memref<64x128xf16>) {
  %m = arith.constant 32 : i32
  %n = arith.constant 128 : i32
  %k = arith.constant 64 : i32

  %out_ptr_idx = memref.extract_aligned_pointer_as_index %out : memref<32x128xf16> -> index
  %out_ptr_i64 = arith.index_cast %out_ptr_idx : index to i64
  %out_ptr = llvm.inttoptr %out_ptr_i64 : i64 to !llvm.ptr

  %act_ptr_idx = memref.extract_aligned_pointer_as_index %act : memref<32x64xf16> -> index
  %act_ptr_i64 = arith.index_cast %act_ptr_idx : index to i64
  %act_ptr = llvm.inttoptr %act_ptr_i64 : i64 to !llvm.ptr

  %wgt_ptr_idx = memref.extract_aligned_pointer_as_index %wgt : memref<64x128xf16> -> index
  %wgt_ptr_i64 = arith.index_cast %wgt_ptr_idx : index to i64
  %wgt_ptr = llvm.inttoptr %wgt_ptr_i64 : i64 to !llvm.ptr

  // CHECK-LABEL: func.func @macro_mm_f16
  // CHECK: llvm.call @hexkl_macro_mm_f16({{.*}}) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  hexkl.macro_mm_f16(%m, %n, %k, %out_ptr, %act_ptr, %wgt_ptr) : i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr
  return
}

// qurt_mem_cache_clean: (i32, i32, i32, i32) -> i32
func.func @mem_cache_clean() {
  %ptr_i32 = arith.constant 1024 : i32
  %size_bytes_i32 = arith.constant 4096 : i32
  %op = arith.constant 0 : i32
  %cache = arith.constant 1 : i32
  // CHECK-LABEL: func.func @mem_cache_clean
  // CHECK: llvm.call @qurt_mem_cache_clean({{.*}}) : (i32, i32, i32, i32) -> i32
  hexkl.qurt_mem_cache_clean(%ptr_i32, %size_bytes_i32, %op, %cache) : i32, i32, i32, i32
  return
}
