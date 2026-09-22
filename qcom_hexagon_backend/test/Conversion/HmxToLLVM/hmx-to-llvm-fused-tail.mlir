//===- hmx-to-llvm-fused-tail.mlir - fused tail to the address-only ABI --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The fused tail (`hmx.unpack_acc_f32`) is one leaf call per row-pair, or one
// ranged call (`hmx_unpack_acc_f32_bulk`) covering a whole tile row when the op
// carries a `count`:
// `hmx_unpack_acc_f32(dst, res, has_res, crouton, rows, cols, dst_stride,
// res_stride, row, block)`.
// An absent residual lowers to address 0 and flag 0, so both op forms share
// one symbol and the wiring phase can thread C through without a second op.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The leaf is declared as llvm.func, never func.func: the latter would wrap
// the symbol in an _mlir_ciface_* shim and the .so would not resolve it.
// CHECK-DAG: llvm.func @hmx_unpack_acc_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// The lock spans the whole kernel: one ensure at entry, one unlock at exit.
// CHECK-DAG: llvm.func @hexagon_runtime_hmx_ensure_dsp()
// CHECK-DAG: llvm.func @hexagon_runtime_hmx_unlock_dsp()

// CHECK-LABEL: func.func @tail_plain
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// dst address = dst_base + descriptor_offset * F32_ESZ (4 bytes/f32).
// CHECK: %[[DST_OFF:.*]] = llvm.trunc
// CHECK: %[[F32_ESZ:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[DST_OFFB:.*]] = llvm.mul %[[DST_OFF]], %[[F32_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[DST_ADDR:.*]] = llvm.add %[[DST_BASE:.*]], %[[DST_OFFB]]
// CHECK-SAME: : i32
// No residual: address 0, flag 0, residual stride 0.
// CHECK: %[[RES_ADDR:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[HAS_RES:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[RES_STRIDE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ZERO_ADD:.*]] = llvm.mlir.constant(0 : i32)
// crouton address = crouton_base + descriptor_offset * CR_ESZ
//                   + (row * ROW_STRIDE + 0) * CR_BYTES.
// CHECK: %[[CR_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[CR_OFFB:.*]] = llvm.mul %[[CR_OFF:.*]], %[[CR_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BASE:.*]] = llvm.add %[[CR_PTR:.*]], %[[CR_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[ROW_STRIDE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[ROW_I64:.*]] = llvm.trunc
// CHECK: %[[ROW_OFF:.*]] = llvm.mul %[[ROW_I64]], %[[ROW_STRIDE]]
// CHECK-SAME: : i32
// CHECK: %[[ROW_PLUS:.*]] = llvm.add %[[ROW_OFF]], %[[ZERO_ADD]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[CR_TILE:.*]] = llvm.mul %[[ROW_PLUS]], %[[CR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[CR_ADDR:.*]] = llvm.add %[[CR_BASE]], %[[CR_TILE]]
// CHECK-SAME: : i32
// rows = 64, cols = 32, dst_stride = 32 (the f32 row stride); row, block last.
// CHECK: %[[ROWS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[ROW_ARG:.*]] = llvm.trunc
// CHECK: %[[BLOCK_ARG:.*]] = llvm.trunc
// CHECK: llvm.call @hmx_unpack_acc_f32(%[[DST_ADDR]], %[[RES_ADDR]], %[[HAS_RES]], %[[CR_ADDR]], %[[ROWS]], %[[COLS]], %[[COLS]], %[[RES_STRIDE]], %[[ROW_ARG]], %[[BLOCK_ARG]]) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @tail_plain(%ar: memref<2x1x16x32x2xf16, 1>,
                      %dst32: memref<64x32xf32>,
                      %row: index, %col: index) {
  hmx.unpack_acc_f32 ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst32 : memref<64x32xf32>)
  return
}

// CHECK-LABEL: func.func @tail_residual
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// dst address = dst_base + descriptor_offset * F32_ESZ (4 bytes/f32).
// CHECK: %[[DST_OFF:.*]] = llvm.trunc
// CHECK: %[[F32_ESZ:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[DST_OFFB:.*]] = llvm.mul %[[DST_OFF]], %[[F32_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[DST_ADDR:.*]] = llvm.add %[[DST_BASE:.*]], %[[DST_OFFB]]
// CHECK-SAME: : i32
// residual address = res_base + descriptor_offset * RES_ESZ (4 bytes/f32).
// CHECK: %[[RES_OFF:.*]] = llvm.trunc
// CHECK: %[[RES_ESZ:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[RES_OFFB:.*]] = llvm.mul %[[RES_OFF]], %[[RES_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[RES_ADDR:.*]] = llvm.add %[[RES_BASE:.*]], %[[RES_OFFB]]
// CHECK-SAME: : i32
// Residual present: flag 1, residual stride 32 (the f32 row stride).
// CHECK: %[[HAS_RES:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[RES_STRIDE:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[ZERO_ADD:.*]] = llvm.mlir.constant(0 : i32)
// crouton address = crouton_base + descriptor_offset * CR_ESZ
//                   + (row * ROW_STRIDE + 0) * CR_BYTES.
// CHECK: %[[CR_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[CR_OFFB:.*]] = llvm.mul %[[CR_OFF:.*]], %[[CR_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BASE:.*]] = llvm.add %[[CR_PTR:.*]], %[[CR_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[ROW_STRIDE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[ROW_I64:.*]] = llvm.trunc
// CHECK: %[[ROW_OFF:.*]] = llvm.mul %[[ROW_I64]], %[[ROW_STRIDE]]
// CHECK-SAME: : i32
// CHECK: %[[ROW_PLUS:.*]] = llvm.add %[[ROW_OFF]], %[[ZERO_ADD]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[CR_TILE:.*]] = llvm.mul %[[ROW_PLUS]], %[[CR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[CR_ADDR:.*]] = llvm.add %[[CR_BASE]], %[[CR_TILE]]
// CHECK-SAME: : i32
// rows = 64, cols = 32, dst_stride = 32, res_stride = 32; row, block last.
// CHECK: %[[ROWS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[ROW_ARG:.*]] = llvm.trunc
// CHECK: %[[BLOCK_ARG:.*]] = llvm.trunc
// CHECK: llvm.call @hmx_unpack_acc_f32(%[[DST_ADDR]], %[[RES_ADDR]], %[[HAS_RES]], %[[CR_ADDR]], %[[ROWS]], %[[COLS]], %[[COLS]], %[[RES_STRIDE]], %[[ROW_ARG]], %[[BLOCK_ARG]]) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @tail_residual(%ar: memref<2x1x16x32x2xf16, 1>,
                         %dst32: memref<64x32xf32>,
                         %res: memref<64x32xf32>,
                         %row: index, %col: index) {
  hmx.unpack_acc_f32 ins(%ar, %row, %col, %res : memref<2x1x16x32x2xf16, 1>, memref<64x32xf32>)
      outs(%dst32 : memref<64x32xf32>)
  return
}

// -----

// A `count` selects the ranged fused tail: the count *replaces* the single
// entry's `col` (the leaf walks the crouton row itself, so `col` was always 0),
// so the call signature stays ten i32 wide. The count is passed as the last
// operand.
// CHECK-LABEL: func.func @tail_ranged
// CHECK: llvm.call @hmx_unpack_acc_f32_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @tail_ranged(%ar: memref<2x1x16x32x2xf16, 1>,
                       %dst32: memref<64x32xf32>,
                       %row: index, %col: index) {
  hmx.unpack_acc_f32 ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst32 : memref<64x32xf32>) {count = 8 : i64}
  return
}
