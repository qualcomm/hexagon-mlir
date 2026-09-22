//===- nsplit-store.mlir - N-split store keeps the row stride -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A block of an output that is split along N is stored into a wider matrix:
// its memref shape is [M, BN] but its row stride is N. `tt.store` of a
// block_ptr becomes `bufferization.materialize_in_destination` into exactly
// such a strided view, and one-shot bufferization forwards that destination
// straight into the HMX read-out, so `hmx.unpack_acc` gets a strided `dst`.
//
// The read-out leaf addresses destination rows by the *row stride*, not by the
// block width. Here the leaf contract must therefore carry 512 (N) next to the
// 256-wide block; passing only the width writes M*BN contiguous elements at the
// block's column offset, which is why an N-split store today keeps only the
// first M*BN elements of the output.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The destination is an N-block of a 32x512 row-major matrix: width 256,
// row stride 512. The unpack must receive rows, width and stride separately.
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// CHECK: %[[ROWS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(256 : i32)
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_unpack_acc_f16({{.*}}, %[[ROWS]], %[[COLS]], %[[STRIDE]], {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @nsplit_store(%ar: memref<1x2x16x32x2xf16, 1>,
                        %dst: memref<32x256xf16, strided<[512, 1]>>,
                        %row: index, %col: index) {
  hmx.unpack_acc ins(%ar, %row, %col : memref<1x2x16x32x2xf16, 1>)
      outs(%dst : memref<32x256xf16, strided<[512, 1]>>)
  return
}

// -----

// A narrow N block of the same wider matrix: width 64 (one full 64-column
// chunk) and width 96 (one full chunk plus a 32-column remainder). Both are
// below the old 256-column unroll threshold, and the leaf reaches its
// register-resident / aligned-store path for every full chunk regardless, so
// the width must keep travelling separately from the row stride here too.
// CHECK-LABEL: func.func @nsplit_store_narrow
// CHECK: %[[NROWS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[NCOLS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[NSTRIDE:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_unpack_acc_f16({{.*}}, %[[NROWS]], %[[NCOLS]], %[[NSTRIDE]], {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @nsplit_store_narrow(%ar: memref<1x2x16x32x2xf16, 1>,
                               %dst: memref<32x64xf16, strided<[512, 1]>>,
                               %row: index, %col: index) {
  hmx.unpack_acc ins(%ar, %row, %col : memref<1x2x16x32x2xf16, 1>)
      outs(%dst : memref<32x64xf16, strided<[512, 1]>>)
  return
}

// CHECK-LABEL: func.func @nsplit_store_partial
// CHECK: %[[PROWS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[PCOLS:.*]] = llvm.mlir.constant(96 : i32)
// CHECK: %[[PSTRIDE:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_unpack_acc_f16({{.*}}, %[[PROWS]], %[[PCOLS]], %[[PSTRIDE]], {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @nsplit_store_partial(%ar: memref<1x2x16x32x2xf16, 1>,
                                %dst: memref<32x96xf16, strided<[512, 1]>>,
                                %row: index, %col: index) {
  hmx.unpack_acc ins(%ar, %row, %col : memref<1x2x16x32x2xf16, 1>)
      outs(%dst : memref<32x96xf16, strided<[512, 1]>>)
  return
}
