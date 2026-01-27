// RUN: linalg-hexagon-opt %s -hexkl-to-llvm | FileCheck %s

// This test checks that each HexKL micro op lowers to an LLVM call with the
// expected extern C signature.

// Setup acc read: (ptr, i32)
func.func @setup_acc_read(%hmx: memref<?xi8, 1>) {
  // CHECK-LABEL: func.func @setup_acc_read
  // CHECK: llvm.call @hexkl_micro_hmx_setup_acc_read_f16({{.*}}) : (!llvm.ptr, i32) -> ()
  hexkl.micro_hmx_setup_acc_read_f16(%hmx) : memref<?xi8, 1>
  return
}

// Acc clear: ()
func.func @acc_clear() {
  // CHECK-LABEL: func.func @acc_clear
  // CHECK: llvm.call @hexkl_micro_hmx_acc_clear_f16() : () -> ()
  hexkl.micro_hmx_acc_clear_f16
  return
}

// Acc read: (ptr, i32, i32) – config offset derived in lowering, out_offset is operand
func.func @acc_read(%hmx: memref<?xi8, 1>) {
  %out_off = arith.constant 8192 : i32
  // CHECK-LABEL: func.func @acc_read
  // CHECK: llvm.call @hexkl_micro_hmx_acc_read_f16({{.*}}) : (!llvm.ptr, i32, i32) -> ()
  hexkl.micro_hmx_acc_read_f16(%hmx, %out_off) : memref<?xi8, 1>, i32
  return
}

// Copy submatrix to f16: (ptr, i32, ptr, i32, i32, i32, i32)
func.func @copy_submatrix_to_f16(%hmx: memref<?xi8, 1>, %src: memref<64x64xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %row = arith.constant 0 : i32
  %col = arith.constant 0 : i32
  %rows = arith.constant 64 : i32
  %cols = arith.constant 64 : i32
  // CHECK-LABEL: func.func @copy_submatrix_to_f16
  // CHECK: llvm.call @hexkl_micro_hmx_copy_submatrix_to_f16({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32) -> ()
  hexkl.micro_hmx_copy_submatrix_to_f16(%hmx, %c0_i32, %src, %row, %col, %rows, %cols) : memref<?xi8, 1>, i32, memref<64x64xf16>, i32, i32, i32, i32
  return
}

// rm_to_ah_f16: (ptr, i32, i32)
func.func @rm_to_ah(%hmx: memref<?xi8, 1>) {
  %act_out = arith.constant 0 : i32
  %flat_in = arith.constant 2048 : i32
  // CHECK-LABEL: func.func @rm_to_ah
  // CHECK: llvm.call @hexkl_micro_hmx_rm_to_ah_f16({{.*}}) : (!llvm.ptr, i32, i32) -> ()
  hexkl.micro_hmx_rm_to_ah_f16(%hmx, %act_out, %flat_in) : memref<?xi8, 1>, i32, i32
  return
}

// rm_to_wh_f16: (ptr, i32, ptr, i32, i32, i32)
func.func @rm_to_wh(%hmx: memref<?xi8, 1>, %wt: memref<64x64xf16>) {
  %wt_off = arith.constant 0 : i32
  %row = arith.constant 0 : i32
  %col = arith.constant 0 : i32
  %wt_cols = arith.constant 64 : i32
  // CHECK-LABEL: func.func @rm_to_wh
  // CHECK: llvm.call @hexkl_micro_hmx_rm_to_wh_f16({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32) -> ()
  hexkl.micro_hmx_rm_to_wh_f16(%hmx, %wt_off, %wt, %row, %col, %wt_cols) : memref<?xi8, 1>, i32, memref<64x64xf16>, i32, i32, i32
  return
}

// mm_f16: (ptr, i32, i32)
func.func @mm_f16(%hmx: memref<?xi8, 1>) {
  %act_off = arith.constant 0 : i32
  %wt_off = arith.constant 4096 : i32
  // CHECK-LABEL: func.func @mm_f16
  // CHECK: llvm.call @hexkl_micro_hmx_mm_f16({{.*}}) : (!llvm.ptr, i32, i32) -> ()
  hexkl.micro_hmx_mm_f16(%hmx, %act_off, %wt_off) : memref<?xi8, 1>, i32, i32
  return
}

// ah_to_rm_f16: (ptr, i32, i32)
func.func @ah_to_rm(%hmx: memref<?xi8, 1>) {
  %flat_out = arith.constant 10240 : i32
  %act_in = arith.constant 8192 : i32
  // CHECK-LABEL: func.func @ah_to_rm
  // CHECK: llvm.call @hexkl_micro_hmx_ah_to_rm_f16({{.*}}) : (!llvm.ptr, i32, i32) -> ()
  hexkl.micro_hmx_ah_to_rm_f16(%hmx, %flat_out, %act_in) : memref<?xi8, 1>, i32, i32
  return
}

// copy_f16_to_f32_submatrix: (ptr, i32, ptr, i32, i32, i32, i32)
func.func @copy_f16_to_f32(%hmx: memref<?xi8, 1>, %dst: memref<64x64xf32>) {
  %in_off = arith.constant 10240 : i32
  %row = arith.constant 0 : i32
  %col = arith.constant 0 : i32
  %rows = arith.constant 64 : i32
  %cols = arith.constant 64 : i32
  // CHECK-LABEL: func.func @copy_f16_to_f32
  // CHECK: llvm.call @hexkl_micro_hmx_copy_f16_to_f32_submatrix({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32) -> ()
  hexkl.micro_hmx_copy_f16_to_f32_submatrix(%hmx, %in_off, %dst, %row, %col, %rows, %cols) : memref<?xi8, 1>, i32, memref<64x64xf32>, i32, i32, i32, i32
  return
}
