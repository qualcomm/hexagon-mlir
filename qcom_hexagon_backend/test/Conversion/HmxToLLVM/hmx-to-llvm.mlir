//===- hmx-to-llvm.mlir - hmx dialect to the address-only runtime ABI -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The runtime leaves take addresses, so a crouton array plus tile indices becomes
// an integer address, and `hmx.acc_read` becomes two calls because the bias
// registers have to be loaded for the selected set before the read-out.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The leaves are declared as llvm.func, never func.func: the latter would wrap
// the symbol in an _mlir_ciface_* shim and the .so would not resolve it.
// CHECK-DAG: llvm.func @hmx_bias_init_unit_f16(i32)
// CHECK-DAG: llvm.func @hmx_acc_clear_f16()
// CHECK-DAG: llvm.func @hmx_mma_f16(i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_bias_load_f16(i32, i32)
// CHECK-DAG: llvm.func @hmx_acc_store_f16(i32, i32)
// The lock spans the whole kernel: one ensure at entry, one unlock at exit.
// CHECK-DAG: llvm.func @hexagon_runtime_hmx_ensure_dsp()
// CHECK-DAG: llvm.func @hexagon_runtime_hmx_unlock_dsp()

// CHECK-LABEL: func.func @tile
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp() : () -> ()
// bias_init address = bias_base + descriptor_offset * BIAS_ELEM (1 byte/xi8).
// CHECK: llvm.ptrtoint
// CHECK: %[[BIAS_OFF:.*]] = llvm.trunc
// CHECK: %[[BIAS_ELEM:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[BIAS_OFFB:.*]] = llvm.mul %[[BIAS_OFF]], %[[BIAS_ELEM]]
// CHECK-SAME: : i32
// CHECK: %[[BIAS_ADDR:.*]] = llvm.add %[[BIAS_BASE:.*]], %[[BIAS_OFFB]]
// CHECK-SAME: : i32
// CHECK: llvm.call @hmx_bias_init_unit_f16(%[[BIAS_ADDR]]) : (i32) -> ()
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// Activation address = act_base + descriptor_offset * CR_ESZ
//                      + (m * KT + k) * CR_BYTES     (KT = 4, CR_BYTES = 2048).
// CHECK: %[[ACT_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[ACT_OFFB:.*]] = llvm.mul %[[ACT_OFF:.*]], %[[ACT_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_BASE:.*]] = llvm.add %[[ACT_PTR:.*]], %[[ACT_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[KT:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[ACT_M:.*]] = llvm.trunc
// CHECK: %[[M_KT:.*]] = llvm.mul %[[ACT_M]], %[[KT]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_K:.*]] = llvm.trunc
// CHECK: %[[MK_K:.*]] = llvm.add %[[M_KT]], %[[ACT_K]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[ACT_TILE:.*]] = llvm.mul %[[MK_K]], %[[CR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_ADDR:.*]] = llvm.add %[[ACT_BASE]], %[[ACT_TILE]]
// CHECK-SAME: : i32
// Weight address = wt_base + descriptor_offset * WT_ESZ
//                  + (n * KT + k) * WT_BYTES      (KT = 4, WT_BYTES = 2048).
// The weight grid is [Nt, Kt] (stored Wᵀ), so dim0 = N, dim1 = K: the tile
// index pair is (n_tile, k_tile) and the dim0 stride is the K-tile count.
// CHECK: %[[WT_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WT_OFFB:.*]] = llvm.mul %[[WT_OFF:.*]], %[[WT_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[WT_BASE:.*]] = llvm.add %[[WT_PTR:.*]], %[[WT_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[WKT:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[WT_N:.*]] = llvm.trunc
// CHECK: %[[N_KT:.*]] = llvm.mul %[[WT_N]], %[[WKT]]
// CHECK-SAME: : i32
// CHECK: %[[WT_K:.*]] = llvm.trunc
// CHECK: %[[NK_K:.*]] = llvm.add %[[N_KT]], %[[WT_K]]
// CHECK-SAME: : i32
// CHECK: %[[WT_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[WT_TILE:.*]] = llvm.mul %[[NK_K]], %[[WT_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[WT_ADDR:.*]] = llvm.add %[[WT_BASE]], %[[WT_TILE]]
// CHECK-SAME: : i32
// CHECK: %[[NCR:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: llvm.call @hmx_mma_f16(%[[ACT_ADDR]], %[[WT_ADDR]], %[[NCR]]) : (i32, i32, i32) -> ()
// acc_read loads bias_set 2 first: bias_load address = bias_base + offset*1.
// CHECK: %[[BIAS_SET:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: llvm.ptrtoint
// CHECK: %[[BIAS_OFF2:.*]] = llvm.trunc
// CHECK: %[[BIAS_ELEM2:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[BIAS_OFFB2:.*]] = llvm.mul %[[BIAS_OFF2]], %[[BIAS_ELEM2]]
// CHECK-SAME: : i32
// CHECK: %[[BIAS_ADDR2:.*]] = llvm.add %[[BIAS_BASE2:.*]], %[[BIAS_OFFB2]]
// CHECK-SAME: : i32
// Then the result store: ar_base + offset*AR_ESZ + (m*NT2 + n)*AR_BYTES.
// CHECK: %[[AR_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[AR_OFFB:.*]] = llvm.mul %[[AR_OFF:.*]], %[[AR_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[AR_BASE:.*]] = llvm.add %[[AR_PTR:.*]], %[[AR_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[NT2:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[AR_M:.*]] = llvm.trunc
// CHECK: %[[M_NT:.*]] = llvm.mul %[[AR_M]], %[[NT2]]
// CHECK-SAME: : i32
// CHECK: %[[AR_N:.*]] = llvm.trunc
// CHECK: %[[MN_N:.*]] = llvm.add %[[M_NT]], %[[AR_N]]
// CHECK-SAME: : i32
// CHECK: %[[AR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[AR_TILE:.*]] = llvm.mul %[[MN_N]], %[[AR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[AR_ADDR:.*]] = llvm.add %[[AR_BASE]], %[[AR_TILE]]
// CHECK-SAME: : i32
// CHECK: llvm.call @hmx_bias_load_f16(%[[BIAS_ADDR2]], %[[BIAS_SET]]) : (i32, i32) -> ()
// CHECK: llvm.call @hmx_acc_store_f16(%[[AR_ADDR]], %[[BIAS_SET]]) : (i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp() : () -> ()
func.func @tile(%bias: memref<256xi8, 1>, %act: memref<2x4x16x32x2xf16, 1>,
                %wt: memref<2x4x16x32x2xf16, 1>,
                %ar: memref<2x2x16x32x2xf16, 1>,
                %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32}
      : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
  return
}

// -----

// The crouton bridge is one runtime leaf per crouton (or one ranged leaf when
// the op carries a `count`): the layout work the engine needs is a vectorised
// pack, not a general transpose. Without this the bridge lowers into element-wise
// moves that measured at 99.3% of the kernel's runtime.
// CHECK-LABEL: func.func @bridge
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// Activation pack address = act_base + descriptor_offset * CR_ESZ
//                           + (row * COL_STRIDE + col) * CR_BYTES.
// COL_STRIDE = 2 is the crouton-count row stride; CR_BYTES = 2048.
// CHECK: %[[ACT_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[ACT_OFFB:.*]] = llvm.mul %[[ACT_OFF:.*]], %[[ACT_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_BASE:.*]] = llvm.add %[[ACT_PTR:.*]], %[[ACT_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[COL_STRIDE:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[ACT_ROW:.*]] = llvm.trunc
// CHECK: %[[ROW_OFF:.*]] = llvm.mul %[[ACT_ROW]], %[[COL_STRIDE]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_COL:.*]] = llvm.trunc
// CHECK: %[[ROWCOL:.*]] = llvm.add %[[ROW_OFF]], %[[ACT_COL]]
// CHECK-SAME: : i32
// CHECK: %[[CR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[ACT_TILE:.*]] = llvm.mul %[[ROWCOL]], %[[CR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_ADDR:.*]] = llvm.add %[[ACT_BASE]], %[[ACT_TILE]]
// CHECK-SAME: : i32
// CHECK: %[[SRC_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[SRC_OFFB:.*]] = llvm.mul %[[SRC_OFF:.*]], %[[SRC_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[SRC_BASE:.*]] = llvm.add %[[SRC_PTR:.*]], %[[SRC_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[ACT_ROWS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[ACT_COLS:.*]] = llvm.mlir.constant(64 : i32)
// The dense source has stride(0) == cols, so the row stride argument reuses
// `ACT_COLS`; a strided source would emit its own constant here.
// CHECK: %[[ACT_ROW2:.*]] = llvm.trunc
// CHECK: %[[ACT_COL2:.*]] = llvm.trunc
// CHECK: llvm.call @hmx_pack_act_f16(%[[ACT_ADDR]], %[[SRC_BASE]], %[[ACT_ROWS]], %[[ACT_COLS]], %[[ACT_COLS]], %[[ACT_ROW2]], %[[ACT_COL2]]) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// Weight pack address = wt_base + offset * WT_ESZ + (nt * KT + kt) * WT_BYTES,
// with KT = 2. The weight grid is [Nt, Kt] (Wᵀ), so dim0 = N carries the K-tile
// stride and dim1 = K is contiguous: the tile index pair is (n_tile, k_tile).
// CHECK: %[[WT_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WT_OFFB:.*]] = llvm.mul %[[WT_OFF:.*]], %[[WT_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[WT_BASE:.*]] = llvm.add %[[WT_PTR:.*]], %[[WT_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[WKT:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WT_NTI:.*]] = llvm.trunc
// CHECK: %[[NT_OFF:.*]] = llvm.mul %[[WT_NTI]], %[[WKT]]
// CHECK-SAME: : i32
// CHECK: %[[WT_KT:.*]] = llvm.trunc
// CHECK: %[[NTKT:.*]] = llvm.add %[[NT_OFF]], %[[WT_KT]]
// CHECK-SAME: : i32
// CHECK: %[[WT_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[WT_TILE:.*]] = llvm.mul %[[NTKT]], %[[WT_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[WT_ADDR:.*]] = llvm.add %[[WT_BASE]], %[[WT_TILE]]
// CHECK-SAME: : i32
// CHECK: %[[WSRC_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WSRC_OFFB:.*]] = llvm.mul %[[WSRC_OFF:.*]], %[[WSRC_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[WSRC_BASE:.*]] = llvm.add %[[WSRC_PTR:.*]], %[[WSRC_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[W_ROWS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[W_COLS:.*]] = llvm.mlir.constant(32 : i32)
// Dense source (`memref<64x32xf16>`): stride(0) == cols, so the row stride
// argument is `W_COLS` again; a strided source emits its own constant.
// CHECK: %[[W_KT:.*]] = llvm.trunc
// CHECK: %[[W_NT:.*]] = llvm.trunc
// CHECK: llvm.call @hmx_pack_weight_f16(%[[WT_ADDR]], %[[WSRC_BASE]], %[[W_ROWS]], %[[W_COLS]], %[[W_COLS]], %[[W_KT]], %[[W_NT]]) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// Unpack address = ar_base + offset * AR_ESZ + (row * ROW_STRIDE + 0) * AR_BYTES;
// the accumulator row stride is one crouton (ROW_STRIDE = 1).
// CHECK: %[[DST_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[DST_OFFB:.*]] = llvm.mul %[[DST_OFF:.*]], %[[DST_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[DST_BASE:.*]] = llvm.add %[[DST_PTR:.*]], %[[DST_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[ZERO_COL:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[AR_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[AR_OFFB:.*]] = llvm.mul %[[AR_OFF:.*]], %[[AR_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[AR_BASE:.*]] = llvm.add %[[AR_PTR:.*]], %[[AR_OFFB]]
// CHECK-SAME: : i32
// CHECK: %[[ROW_STRIDE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[AR_ROW:.*]] = llvm.trunc
// CHECK: %[[AR_ROWOFF:.*]] = llvm.mul %[[AR_ROW]], %[[ROW_STRIDE]]
// CHECK-SAME: : i32
// CHECK: %[[AR_ROWCOL:.*]] = llvm.add %[[AR_ROWOFF]], %[[ZERO_COL]]
// CHECK-SAME: : i32
// CHECK: %[[AR_BYTES:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[AR_TILE:.*]] = llvm.mul %[[AR_ROWCOL]], %[[AR_BYTES]]
// CHECK-SAME: : i32
// CHECK: %[[AR_ADDR:.*]] = llvm.add %[[AR_BASE]], %[[AR_TILE]]
// CHECK-SAME: : i32
// CHECK: %[[OUT_ROWS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[OUT_COLS:.*]] = llvm.mlir.constant(32 : i32)
// The destination is dense (`memref<64x32xf16>`), so stride(0) == cols and the
// lowering reuses the width as the row stride: the 5th argument is `OUT_COLS`
// again. A strided destination (an N-split store) would emit its own constant.
// CHECK: %[[OUT_ROW:.*]] = llvm.trunc
// CHECK: %[[OUT_COL:.*]] = llvm.trunc
// CHECK: llvm.call @hmx_unpack_acc_f16(%[[DST_BASE]], %[[AR_ADDR]], %[[OUT_ROWS]], %[[OUT_COLS]], %[[OUT_COLS]], %[[OUT_ROW]], %[[OUT_COL]]) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @bridge(%src: memref<64x64xf16>, %wsrc: memref<64x32xf16>,
                  %act: memref<2x2x16x32x2xf16, 1>,
                  %wt: memref<1x2x16x32x2xf16, 1>,
                  %ar: memref<2x1x16x32x2xf16, 1>,
                  %dst16: memref<64x32xf16>,
                  %row: index, %col: index, %kt: index, %nt: index) {
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x32xf16>)
      outs(%wt : memref<1x2x16x32x2xf16, 1>)
  hmx.unpack_acc ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst16 : memref<64x32xf16>)
  return
}

// A pack source can be a view into the middle of a buffer (the K block of an
// attention matmul): `HexagonPuntBuffer` forwards such a view straight into
// `hmx.pack_weight`, so the descriptor's offset has to be added to the address.
// Dropping it made every K block pack block 0, measured as a wrong f16 attention
// result whose error grew with the number of K blocks. `@bridge` above only uses
// offset 0 and therefore cannot catch it.
// CHECK-LABEL: func.func @bridge_offset
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp(
// CHECK: %[[VDESC:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<64x64xf16, strided<[64, 1], offset: ?>> to
// CHECK: %[[PTR:.*]] = llvm.extractvalue %[[VDESC]][1] : {{.*}}
// CHECK: %[[BASE:.*]] = llvm.ptrtoint %[[PTR]] : {{.*}} to i32
// CHECK: %[[IDX_I64:.*]] = llvm.extractvalue %[[VDESC]][2] : {{.*}}
// CHECK: %[[IDX:.*]] = llvm.trunc %[[IDX_I64]] : i64 to i32
// CHECK: %[[VIEW_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[OFF:.*]] = llvm.mul %[[IDX]], %[[VIEW_ESZ]]
// CHECK-SAME: : i32
// CHECK: %[[ADDR:.*]] = llvm.add %[[BASE]], %[[OFF]]
// CHECK-SAME: : i32
// CHECK: llvm.call @hmx_pack_weight_f16({{.*}}, %[[ADDR]], {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp(
func.func @bridge_offset(%base: memref<128x64xf16>, %o: index,
                         %wt: memref<2x2x16x32x2xf16, 1>, %kt: index,
                         %nt: index) {
  %view = memref.reinterpret_cast %base to offset: [%o], sizes: [64, 64],
      strides: [64, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
  hmx.pack_weight ins(%view, %kt, %nt : memref<64x64xf16, strided<[64, 1], offset: ?>>)
      outs(%wt : memref<2x2x16x32x2xf16, 1>)
  return
}

// -----

// `hmx.stage` / `hmx.await` lower to the existing DMA runtime entries, one
// *unconditional* call each: `row` is an element row offset, so the source
// address is base + row * stride(0) * elemBytes, and the wait takes the token
// the start returned. There is no out-of-range branch and no token sentinel --
// the runtime's real tokens start at 0, so 0 cannot mean "no transfer".
// A function that only stages and awaits issues no HMX instruction, so it needs
// no engine ensure/unlock: the calls are plain DMA. (A kernel that also runs
// `hmx.mma` still gets the pair; see @tile.)
// CHECK-LABEL: func.func @stage_await
// Source address = src base + descriptor offset, then row * 1024 * 2 bytes.
// CHECK: %[[ROW_I64:.*]] = builtin.unrealized_conversion_cast %arg3 : index to i64
// CHECK: %[[ROW:.*]] = llvm.trunc %[[ROW_I64]] : i64 to i32
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1024 : i32)
// CHECK: %[[ROW_STRIDE:.*]] = llvm.mul %[[ROW]], %[[STRIDE]] : i32
// CHECK: %[[ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[ROW_BYTES:.*]] = llvm.mul %[[ROW_STRIDE]], %[[ESZ]] : i32
// CHECK: %[[SRC_BASE:.*]] = llvm.add %{{.*}}, %{{.*}} : i32
// CHECK: %[[SRC_ADDR:.*]] = llvm.add %[[SRC_BASE]], %[[ROW_BYTES]] : i32
// Length = the slot's 32768 elements * 2 bytes.
// CHECK: %[[LEN_ELEMS:.*]] = llvm.mlir.constant(32768 : i32)
// CHECK: %[[LEN:.*]] = llvm.mul %[[LEN_ELEMS]], %{{.*}} : i32
// The source is DDR (space 0), the slot VTCM (space 1), the status a pointer.
// CHECK: %[[SRC_PTR:.*]] = llvm.inttoptr %[[SRC_ADDR]] : i32 to !llvm.ptr
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[STATUS_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[TOKEN:.*]] = llvm.call @hexagon_runtime_dma_start(%[[SRC_PTR]], %[[SRC_SPACE]], %[[DST_PTR]], %[[DST_SPACE]], %[[LEN]], {{.*}}, {{.*}}, %[[STATUS_PTR]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// The wait is unconditional: one call on the token the start returned, with no
// `icmp`/`cf` sentinel branch.
// CHECK: llvm.call @hexagon_runtime_dma_wait(%[[TOKEN]]) : (i32) -> ()
// CHECK-NOT: cf.
// CHECK-NOT: llvm.icmp
// CHECK-NOT: hmx.stage
func.func @stage_await(%src: memref<64x1024xf16>, %slot: memref<32x1024xf16, 1>,
                       %status: memref<1xi32>, %row: index) {
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>) outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  %ready = hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
  return
}

// -----

// A `count` above 1 selects the ranged (bulk) leaf: one call covers the whole K
// run (pack) or the whole tile row (unpack), and the count travels as the extra
// final i32 for the pack -- which is what makes its call eight i32 wide, against
// the seven of the single entry. The unpack count *replaces* the single entry's
// `col` (the leaf walks the crouton row itself, so `col` was always 0), so its
// call stays seven wide. `count` absent keeps the single-crouton entry.
// CHECK-LABEL: func.func @bridge_ranged
// CHECK: llvm.call @hmx_pack_act_f16_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hmx_pack_weight_f16_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// The unpack count (8) is materialised right before its call.
// CHECK: llvm.mlir.constant(8 : i32)
// CHECK: llvm.call @hmx_unpack_acc_f16_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @bridge_ranged(%src: memref<64x64xf16>, %wsrc: memref<64x96xf16>,
                         %act: memref<2x2x16x32x2xf16, 1>,
                         %wt: memref<3x2x16x32x2xf16, 1>,
                         %ar: memref<2x1x16x32x2xf16, 1>,
                         %dst16: memref<64x32xf16>,
                         %row: index, %col: index, %kt: index, %nt: index) {
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>) {count = 2 : i64}
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x96xf16>)
      outs(%wt : memref<3x2x16x32x2xf16, 1>) {count = 2 : i64}
  hmx.unpack_acc ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst16 : memref<64x32xf16>) {count = 8 : i64}
  return
}

// -----

// The source's element type picks the leaf: an f32 source goes to the pack's
// f32 entry, which folds the engine's fp16 conversion into the same pass. The
// call shape is the f16 one unchanged -- seven i32 for the single block, eight
// with the `count` of the ranged form -- and the address is the source's own
// f32 element size, so the leaf's `src_stride` is in f32 elements.
// CHECK-LABEL: func.func @bridge_f32_source
// CHECK: llvm.call @hmx_pack_act_f32({{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hmx_pack_weight_f32_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK: llvm.call @hmx_unpack_acc_f16_bulk({{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT: hmx_pack_act_f16
// CHECK-NOT: hmx_pack_weight_f16
func.func @bridge_f32_source(%src: memref<64x64xf32>, %wsrc: memref<64x96xf32>,
                             %act: memref<2x2x16x32x2xf16, 1>,
                             %wt: memref<3x2x16x32x2xf16, 1>,
                             %ar: memref<2x1x16x32x2xf16, 1>,
                             %dst16: memref<64x32xf16>,
                             %row: index, %col: index, %kt: index, %nt: index) {
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf32>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x96xf32>)
      outs(%wt : memref<3x2x16x32x2xf16, 1>) {count = 2 : i64}
  hmx.unpack_acc ins(%ar, %row, %col : memref<2x1x16x32x2xf16, 1>)
      outs(%dst16 : memref<64x32xf16>) {count = 8 : i64}
  return
}
