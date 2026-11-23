//===- decompose-hexkl-matmul-to-daemon.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(inject-hmx-thread-id-arg,inject-hmx-vtcm-base-arg,inject-hmx-queue-arg,decompose-hexkl-matmul-to-daemon))' \
// RUN:   | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(inject-hmx-thread-id-arg,inject-hmx-vtcm-base-arg,inject-hmx-queue-arg,decompose-hexkl-matmul-to-daemon))' \
// RUN:   | FileCheck %s --check-prefix=NOACC

// decompose-hexkl-matmul-to-daemon lowers hexkl.matmul into WORKER-side layout
// ops plus hexkl.micro_hmx_dispatch (one K=32 job) and one per-output-tile
// hexkl.micro_hmx_wait_idle.  The single HMX daemon thread owns the
// accumulator, so the worker IR must NOT contain setup_acc_read / acc_clear /
// mm / acc_read, and must NOT allocate VTCM: it uses the injected global VTCM
// base instead.  The three inject passes run first so the func tail is
// [..., %tid: i32, %vtcm: memref<?xi8, 1>, %queue: !llvm.ptr] with the matching
// hexkl_hmx.vtcm_base_arg / hmx.{thread_id,queue}_arg func attrs the pass
// reads.  Exact VTCM offset values are validated by the end-to-end numeric diff
// against hexKLMode="micro"; here we lock the op sequence, nesting, and types.

// -----

// Static single tile (32x32x32), kTiles == 1.
func.func @test_single_tile(%lhs: memref<32x32xf16>,
                            %rhs: memref<32x32xf16>,
                            %out: memref<32x32xf32>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x32xf16>, memref<32x32xf16>)
               outs(%out : memref<32x32xf32>)
  return
}

// CHECK-LABEL: func.func @test_single_tile
// Injected trailing args: thread-id (i32), VTCM base (memref), HMX queue (ptr).
// CHECK-SAME:  i32
// CHECK-SAME:  memref<?xi8, 1 : i32>
// CHECK-SAME:  !llvm.ptr
// K-independent VTCM layout, emitted as compile-time constants: 2048 B tile
// stride, ring mask 7 (the ring index is kt & 7, a bitwise mask -- never a
// div/mod), wgtBase 16384, flat 32768, acc 34816, scr 36864, per-instance 38912.
// CHECK-DAG:   arith.constant 2048 : i32
// CHECK-DAG:   arith.constant 7 : i32
// CHECK-DAG:   arith.constant 16384 : i32
// CHECK-DAG:   arith.constant 32768 : i32
// CHECK-DAG:   arith.constant 34816 : i32
// CHECK-DAG:   arith.constant 36864 : i32
// CHECK-DAG:   arith.constant 38912 : i32
// M-outer / N-middle / K-inner: three nested scf.for, none carrying iter_args
// (each header ends in `step %.. {`); all offsets are recomputed from the IVs.
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// Ring slot p = kt & 7 (bitwise mask), then actOff = p * 2048 -- no div/mod.
// CHECK:             arith.andi %{{.*}}, %c7_i32
// CHECK:             arith.muli
// All layout lives INSIDE the K-loop (no row prologue): activation copy + AH
// pack, weight WH pack, then one K=32 dispatch.
// CHECK:             hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK:             hexkl.micro_hmx_rm_to_ah_f16
// CHECK:             hexkl.micro_hmx_rm_to_wh_f16
// clear_acc / write_out are runtime arith.cmpi values, not constants.
// CHECK:             arith.cmpi
// CHECK:             hexkl.micro_hmx_dispatch({{.*}}) : !llvm.ptr, i32, i32, i32, i32, i1, i1 {job_kind = "f16"}
// One wait_idle AFTER the K-loop (sibling in the N-loop body), then readback.
// CHECK:           hexkl.micro_hmx_wait_idle({{.*}}) : !llvm.ptr, i32
// CHECK:           hexkl.micro_hmx_ah_to_rm_f16
// CHECK:           hexkl.micro_hmx_copy_f16_to_f32_submatrix

// Worker never touches the HMX accumulator and never allocates VTCM.
// NOACC-LABEL: func.func @test_single_tile
// NOACC-NOT:   hexagonmem.alloc
// NOACC-NOT:   hexkl.micro_hmx_setup_acc_read_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_clear_f16
// NOACC-NOT:   hexkl.micro_hmx_mm_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_read_f16

// -----

// Fully dynamic shapes: pass must still emit dispatch/wait_idle, no accumulator.
func.func @test_dynamic(%lhs: memref<?x?xf16>,
                        %rhs: memref<?x?xf16>,
                        %out: memref<?x?xf32>) {
  hexkl.matmul ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>)
               outs(%out : memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @test_dynamic
// CHECK-SAME:  i32
// CHECK-SAME:  memref<?xi8, 1 : i32>
// CHECK-SAME:  !llvm.ptr
// Even with dynamic K the footprint stays K-independent: same fixed 38912 B
// per-instance layout and the ring index is still kt & 7 (a mask, not a mod).
// CHECK-DAG:   arith.constant 7 : i32
// CHECK-DAG:   arith.constant 38912 : i32
// CHECK:       arith.andi %{{.*}}, %c7_i32
// CHECK:       hexkl.micro_hmx_dispatch({{.*}}) : !llvm.ptr, i32, i32, i32, i32, i1, i1 {job_kind = "f16"}
// CHECK:       hexkl.micro_hmx_wait_idle({{.*}}) : !llvm.ptr, i32

// NOACC-LABEL: func.func @test_dynamic
// NOACC-NOT:   hexkl.micro_hmx_setup_acc_read_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_clear_f16
// NOACC-NOT:   hexkl.micro_hmx_mm_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_read_f16

// -----

// Static deep-K (32x256x32), kTiles == 8.  This is the capability the ring
// unlocks: with K = 256 the VTCM footprint MUST stay the same fixed 38912 B as
// the single-tile case -- the 8 K-blocks cycle through the 8-slot ring (index
// kt & 7) instead of allocating 8 activation + 8 weight buffers.  clear_acc
// fires only on the first K-block and write_out only on the last, so the daemon
// accumulates all 8 blocks into one tile before the single acc_read.
func.func @test_deep_k(%lhs: memref<32x256xf16>,
                       %rhs: memref<256x32xf16>,
                       %out: memref<32x32xf32>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x256xf16>, memref<256x32xf16>)
               outs(%out : memref<32x32xf32>)
  return
}

// CHECK-LABEL: func.func @test_deep_k
// CHECK-SAME:  i32
// CHECK-SAME:  memref<?xi8, 1 : i32>
// CHECK-SAME:  !llvm.ptr
// Footprint is identical to the single-tile case despite K = 256: same fixed
// 38912 B per-instance layout, same ring mask 7 -- buffers do NOT scale with K.
// CHECK-DAG:   arith.constant 7 : i32
// CHECK-DAG:   arith.constant 38912 : i32
// The K-loop runs kTiles == 8 trips (one full sweep of the 8-slot ring); M/N are
// single 32-wide tiles.  No iter_args on any loop.
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// CHECK:           scf.for %{{.*}} = %{{.*}} to %c8 step %{{[^ ]*}} {
// Ring index is still a bitwise mask (kt & 7) across all 8 K-blocks.
// CHECK:             arith.andi %{{.*}}, %c7_i32
// clear_acc only on the first K-block (kt == 0); write_out only on the last
// (kt == kTiles-1 == 7).
// CHECK:             arith.cmpi eq, %{{.*}}, %c0 : index
// CHECK:             arith.cmpi eq, %{{.*}}, %c7 : index
// CHECK:             hexkl.micro_hmx_dispatch({{.*}}) : !llvm.ptr, i32, i32, i32, i32, i1, i1 {job_kind = "f16"}
// One wait_idle after the 8-block K-loop.
// CHECK:           hexkl.micro_hmx_wait_idle({{.*}}) : !llvm.ptr, i32

// NOACC-LABEL: func.func @test_deep_k
// NOACC-NOT:   hexagonmem.alloc
// NOACC-NOT:   hexkl.micro_hmx_setup_acc_read_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_clear_f16
// NOACC-NOT:   hexkl.micro_hmx_mm_f16
// NOACC-NOT:   hexkl.micro_hmx_acc_read_f16

// -----

// Quantized u8(activation) x i8(weight) -> i32, one output tile, kTiles == 1.
//
// The int8 HMX block is NOT the f16 block rescaled by element size: it is 64
// rows deep (HEXKL_HMX_INT8_BLOCK_N_ROW == 64 vs f16's 32), so the M loop steps
// by 64 and tile_row divides by 64.  The three tile sizes all differ --
// activation 64*32*1 = 2048 B, weight 32*32*1 = 1024 B, i32 accumulator
// 64*32*4 = 8192 B -- so the activation and weight rings carry independent
// strides.  Getting any of these wrong is silent: mm_u8i8 would read a
// 2048 B activation tile out of an under-sized slot, and odd ring slots would
// break HEXKL_HMX_ACTIVATION_ALIGNMENT (2048), which the addon documents as
// undefined behavior rather than an error return.
func.func @test_u8i8_single_tile(%lhs: memref<64x32xi8>,
                                 %rhs: memref<32x32xi8>,
                                 %out: memref<64x32xi32>) {
  hexkl.matmul ins(%lhs, %rhs : memref<64x32xi8>, memref<32x32xi8>)
               outs(%out : memref<64x32xi32>)
  return
}

// CHECK-LABEL: func.func @test_u8i8_single_tile
// CHECK-SAME:  i32
// CHECK-SAME:  memref<?xi8, 1 : i32>
// CHECK-SAME:  !llvm.ptr
// u8i8 VTCM layout: activation ring stride 2048 B, weight ring stride 1024 B,
// ring mask 7, wgtBase 16384, acc 24576, per-instance 32768.  There is no flat
// or scr region -- the addon fuses copy+layout and unshuffle+store -- so this
// is exactly 8*2048 + 8*1024 + 8192 = 32768 B, with no unused headroom.
// CHECK-DAG:   arith.constant 2048 : i32
// CHECK-DAG:   arith.constant 1024 : i32
// CHECK-DAG:   arith.constant 7 : i32
// CHECK-DAG:   arith.constant 16384 : i32
// CHECK-DAG:   arith.constant 24576 : i32
// CHECK-DAG:   arith.constant 32768 : i32
// M blocks are 64 rows: the M loop steps by %c64 and tile_row divides by 64.
// CHECK-DAG:   arith.constant 64 : index
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c64 {
// CHECK:         arith.divui %{{.*}}, %c64_i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[^ ]*}} {
// Ring slot p = kt & 7, then act uses the 2048 B stride and wgt the 1024 B one.
// CHECK:             arith.andi %{{.*}}, %c7_i32
// CHECK:             arith.muli %{{.*}}, %c2048_i32
// CHECK:             arith.muli %{{.*}}, %c1024_i32
// Fused DDR->AH activation load (no separate rm_to_ah step), then weight WH.
// CHECK:             hexkl.micro_hmx_copy_submatrix_to_8b_activation
// CHECK:             hexkl.micro_hmx_rm_to_wh_i8
// CHECK:             arith.cmpi
// CHECK:             hexkl.micro_hmx_dispatch({{.*}}) : !llvm.ptr, i32, i32, i32, i32, i1, i1 {job_kind = "u8i8"}
// Readback is a single fused unshuffle+store straight from the accumulator --
// acc_read_int32's output feeds copy_32b_to_submatrix directly, so there is no
// i32 ah_to_rm step (the addon has no such function).
// CHECK:           hexkl.micro_hmx_wait_idle({{.*}}) : !llvm.ptr, i32
// CHECK:           hexkl.micro_hmx_copy_32b_to_submatrix

// Worker never touches the HMX accumulator and never allocates VTCM.  Also
// asserts the u8i8 path does NOT stage through flat/scr buffers (no f16-style
// rm_to_ah / ah_to_rm pair) -- those ops do not exist in the addon's u8i8 API.
// NOACC-LABEL: func.func @test_u8i8_single_tile
// NOACC-NOT:   hexagonmem.alloc
// NOACC-NOT:   hexkl.micro_hmx_rm_to_ah
// NOACC-NOT:   hexkl.micro_hmx_ah_to_rm

// -----

// Quantized deep-K (64x256x32), kTiles == 8: same fixed 32768 B footprint as the
// single-tile u8i8 case, so the ring -- not the K extent -- bounds VTCM here too.
func.func @test_u8i8_deep_k(%lhs: memref<64x256xi8>,
                            %rhs: memref<256x32xi8>,
                            %out: memref<64x32xi32>) {
  hexkl.matmul ins(%lhs, %rhs : memref<64x256xi8>, memref<256x32xi8>)
               outs(%out : memref<64x32xi32>)
  return
}

// CHECK-LABEL: func.func @test_u8i8_deep_k
// CHECK-DAG:   arith.constant 7 : i32
// CHECK-DAG:   arith.constant 32768 : i32
// K = 256 with a 32-wide inner block is 8 trips, one full sweep of the ring.
// CHECK:       scf.for %{{.*}} = %{{.*}} to %c8 step %{{[^ ]*}} {
// CHECK:             arith.andi %{{.*}}, %c7_i32
// clear_acc on the first K-block only, write_out on the last (kt == 7).
// CHECK:             arith.cmpi eq, %{{.*}}, %c0 : index
// CHECK:             arith.cmpi eq, %{{.*}}, %c7 : index
// CHECK:             hexkl.micro_hmx_dispatch({{.*}}) : !llvm.ptr, i32, i32, i32, i32, i1, i1 {job_kind = "u8i8"}
// CHECK:           hexkl.micro_hmx_wait_idle({{.*}}) : !llvm.ptr, i32
// CHECK:           hexkl.micro_hmx_copy_32b_to_submatrix

// NOACC-LABEL: func.func @test_u8i8_deep_k
// NOACC-NOT:   hexagonmem.alloc
