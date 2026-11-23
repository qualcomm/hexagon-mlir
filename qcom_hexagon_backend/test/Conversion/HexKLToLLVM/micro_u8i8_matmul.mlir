//===- micro_u8i8_matmul.mlir ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexkl-to-llvm | FileCheck %s

// U8I8 data movement remains in IR; HMX execution is owned by the daemon.
func.func @u8i8_data_movement(%hmx: memref<?xi8, 1>,
                              %act: memref<64x32xi8>,
                              %wgt: memref<32x32xi8>,
                              %out: memref<64x32xi32>) {
  %c0 = arith.constant 0 : i32
  %rows = arith.constant 64 : i32
  %act_cols = arith.constant 32 : i32
  %wgt_cols = arith.constant 32 : i32
  %act_off = arith.constant 0 : i32
  %wgt_off = arith.constant 2048 : i32
  %out_off = arith.constant 4096 : i32

  // CHECK-LABEL: func.func @u8i8_data_movement
  // CHECK: llvm.call @hexkl_micro_hmx_copy_submatrix_to_8b_activation({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32) -> ()
  hexkl.micro_hmx_copy_submatrix_to_8b_activation(%hmx, %act_off, %act, %c0, %c0, %rows, %act_cols)
      : memref<?xi8, 1>, i32, memref<64x32xi8>, i32, i32, i32, i32
  // CHECK: llvm.call @hexkl_micro_hmx_rm_to_wh_i8({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32) -> ()
  hexkl.micro_hmx_rm_to_wh_i8(%hmx, %wgt_off, %wgt, %c0, %c0, %wgt_cols)
      : memref<?xi8, 1>, i32, memref<32x32xi8>, i32, i32, i32
  // CHECK: llvm.call @hexkl_micro_hmx_copy_32b_to_submatrix({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32) -> ()
  hexkl.micro_hmx_copy_32b_to_submatrix(%hmx, %out_off, %out, %c0, %c0, %rows, %act_cols)
      : memref<?xi8, 1>, i32, memref<64x32xi32>, i32, i32, i32, i32
  return
}
