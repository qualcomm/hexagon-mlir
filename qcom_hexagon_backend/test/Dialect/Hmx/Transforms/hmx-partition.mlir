//===- hmx-partition.mlir - hmx.matmul becomes the tile loop -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The crouton arrays stay memory-space-1 buffers: the VTCM machinery that runs
// after this pass (convert-to-hexagonmem -> the runtime's VTCM pool) is what
// places them, together with every other VTCM buffer of the kernel. So this pass
// only adds the conversion state and the tile loop.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @partition
// The conversion state is one 256-byte block for the whole kernel.
// CHECK: %[[BIAS:.*]] = memref.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
// CHECK: hmx.bias_init %[[BIAS]]
// The crouton buffers are left alone; the tile loop walks them by tile index.
// CHECK: memref.alloc() : memref<2x4x16x32x2xf16, 1>
// CHECK: scf.for %[[M:.*]] = {{.*}} to
// CHECK: scf.for %[[N:.*]] = {{.*}} to
// CHECK: hmx.acc_clear
// CHECK: scf.for %[[K:.*]] = {{.*}} to
// CHECK: hmx.mma %{{.*}}, %{{.*}}, %[[M]], %[[N]], %[[K]] {n_croutons = 1 : i32}
// CHECK: hmx.acc_read %{{.*}}, %{{.*}}, %[[M]], %[[N]] {bias_set = 0 : i32}
// The conversion state has its paired release at the single exit: it is kernel
// setup, not a per-launch leak.
// CHECK: memref.dealloc %[[BIAS]]
// CHECK-NOT: hmx.matmul
func.func @partition() {
  %a = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// Two dots keep their own grids: the partition never unifies tile loops across
// dots. Each matmul lowers to the loop nest its own crouton arrays describe,
// under the single conversion-state block of the kernel.
// CHECK-LABEL: func.func @two_dots_keep_grids
// Only one conversion-state block per kernel.
// CHECK: hmx.bias_init
// CHECK-NOT: hmx.bias_init
// The first dot's grid (Mt=2, Kt=4) ...
// CHECK: arith.constant 4 : index
// CHECK: scf.for
// CHECK: hmx.mma
// CHECK: hmx.acc_read
// ... and the second dot's grid (Mt=8) lowers to its own loop nest.
// CHECK: arith.constant 8 : index
// CHECK: scf.for
// CHECK: hmx.mma
// CHECK: hmx.acc_read
// CHECK-NOT: hmx.matmul
func.func @two_dots_keep_grids() {
  %a0 = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  %w0 = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  %r0 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a0, %w0 : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>)
             outs(%r0 : memref<2x2x16x32x2xf16, 1>)
  %a1 = memref.alloc() : memref<8x2x16x32x2xf16, 1>
  %w1 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %r1 = memref.alloc() : memref<8x2x16x32x2xf16, 1>
  hmx.matmul ins(%a1, %w1 : memref<8x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%r1 : memref<8x2x16x32x2xf16, 1>)
  return
}

// The activation bridge (a pack loop fed by a row-major source) is software-
// pipelined: the pack is re-hosted to read a VTCM slot the DMA filled, and the
// pipelining itself is the SCF pipeliner's, not hand-written. The pass emits
// the serial source loop (issue -> await -> compute per tile) and schedules it
// "issue in stage 0, await and compute in stage 1"; the pipeliner generates the
// prologue (the issue of tile 0), the kernel `issue(m+1) await(m) compute(m)`
// over tiles 0 .. Mt-2, and a peeled epilogue that only awaits and computes
// tile Mt-1. The fixture has Mt=4: the kernel runs three iterations and the
// tail computes the fourth. The (token, slot) pair the await consumes travels
// as the kernel's iter_args -- the pipeliner's versioning is the ring rotation;
// what alternates the two static slots is a parity select over the tile index.
// `row` is an element row offset: tile t starts at source row t * 32. The
// weight stays whole and resident.
//
// The pack destination is one crouton-row scratch, not the whole 4x32 activation
// array: the pack and its mmas are in the same iteration, so one row is live at
// a time and the array is retired whole. The mmas read the scratch at crouton
// row 0; `acc_read` still writes output tile row m.
// CHECK-LABEL: func.func @pipeline
// CHECK: %[[STATE:.*]] = memref.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
// CHECK: hmx.bias_init %[[STATE]]
// The whole activation array is gone: the tile loop fills the scratch instead.
// CHECK-NOT: memref<4x32x16x32x2xf16, 1>
// The weight stays whole and resident: one 2x32x16x32x2 crouton array.
// CHECK: %[[WCRING:.*]] = memref.alloc() : memref<2x32x16x32x2xf16, 1>
// CHECK: hmx.pack_weight ins(%arg1, {{.*}}) outs(%[[WCRING]] :
// The accumulator is one 4x2x16x32x2 crouton array, indexed by the tile row.
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<4x2x16x32x2xf16, 1>
// One crouton-row scratch holds the pack destination: Kt croutons in one row.
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// The activation is staged through a two-slot ring: two 32x1024 f16 slots and
// two status words, each allocated once outside the loop.
// CHECK: %[[SLOT0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// CHECK: %[[ST0:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[SLOT1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// CHECK: %[[ST1:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// Ring/grid constants: depth = 2, tile edge = 32, Kt = 32, Nt = 2, Mt = 4.
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[TILE_EDGE:.*]] = arith.constant 32 : index
// CHECK: %[[KTILE:.*]] = arith.constant 32 : index
// CHECK: %[[NTILE:.*]] = arith.constant 2 : index
// CHECK: %[[MT:.*]] = arith.constant 4 : index
// Prologue (the pipeliner's): the issue part of tile 0 -- parity select of the
// slot, source row 0, and the stage. The select result is what the kernel's
// await consumes one version later.
// CHECK: %[[IV0:.*]] = arith.addi %[[C0]], {{.*}} : index
// CHECK: %[[SSEL0:.*]] = arith.select {{.*}}, %[[SLOT1]], %[[SLOT0]] : memref<32x1024xf16, 1>
// CHECK: %[[SSTSEL0:.*]] = arith.select {{.*}}, %[[ST1]], %[[ST0]] : memref<1xi32>
// CHECK: %[[ROW0:.*]] = arith.muli %[[IV0]], %[[TILE_EDGE]] : index
// CHECK: %[[T0:.*]] = hmx.stage ins(%arg0, %[[ROW0]] : memref<128x1024xf16>) outs(%[[SSEL0]], %[[SSTSEL0]] : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// The kernel stops one iteration short of Mt and carries the (token, slot)
// pair the await consumes as iter_args, initialized from the prologue.
// CHECK: %[[KUB:.*]] = arith.subi %[[MT]], {{.*}} : index
// CHECK: scf.for %[[K:.*]] = %[[C0]] to %[[KUB]] step %[[C1]] iter_args(%[[T:.*]] = %[[T0]], %[[S:.*]] = %[[SSEL0]]) -> (i32, memref<32x1024xf16, 1>) {
// Issue part (stage 0): the pipeliner shifts the induction variable by one
// iteration, so the kernel issues tile m+1 -- source row (m+1) * 32, slot
// parity of m+1 -- before tile m is awaited.
// CHECK: %[[IVK:.*]] = arith.addi %[[K]], {{.*}} : index
// CHECK: %[[SSELK:.*]] = arith.select {{.*}}, %[[SLOT1]], %[[SLOT0]] : memref<32x1024xf16, 1>
// CHECK: %[[ROWK:.*]] = arith.muli {{.*}}, %[[TILE_EDGE]] : index
// CHECK: %[[TNK:.*]] = hmx.stage ins(%arg0, %[[ROWK]] : memref<128x1024xf16>) outs(%[[SSELK]], {{.*}} : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// Compute part (stage 1): the await consumes the versioned token and the slot
// that token's transfer wrote; one ranged pack covers the whole K run into
// destination croutons (0, 0..Kt), and the mmas read the scratch at row 0.
// CHECK: %[[ROW:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[S]] : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// CHECK: hmx.pack_act ins(%[[ROW]], %[[C0]], %[[C0]] : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] : memref<1x32x16x32x2xf16, 1>) {count = 32 : i64}
// CHECK: scf.for %[[N:.*]] = %[[C0]] to %[[NTILE]] step %[[C1]] {
// CHECK: hmx.acc_clear
// CHECK: scf.for %[[K2:.*]] = %[[C0]] to %[[KTILE]] step %[[C1]] {
// CHECK: hmx.mma %[[SCRATCH]], %[[WCRING]], %[[C0]], %[[N]], %[[K2]] {n_croutons = 1 : i32}
// The compute part runs at the unshifted induction variable (the pipeliner
// re-derives it per use, so the m operand is `iv + 0`): output tile m.
// CHECK: hmx.acc_read %[[STATE]], %[[ACC]], {{.*}}, %[[N]] {bias_set = 0 : i32}
// CHECK: scf.yield %[[TNK]], %[[SSELK]] : i32, memref<32x1024xf16, 1>
// Peeled epilogue (the pipeliner's): tile Mt-1 only awaits and computes, from
// the kernel's last (token, slot) results. No stage, so no out-of-range row
// and no token sentinel.
// CHECK: %[[ROWEP:.*]] = hmx.await ins(%{{.*}} : i32) outs(%{{.*}} : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// CHECK: hmx.pack_act ins(%[[ROWEP]], %[[C0]], {{.*}} : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] :
// CHECK: hmx.acc_read {{.*}} {bias_set = 0 : i32}
// The ring and the scratch are released after the epilogue.
// CHECK: memref.dealloc %[[SLOT0]]
// CHECK: memref.dealloc %[[SLOT1]]
// CHECK: memref.dealloc %[[ST0]]
// CHECK: memref.dealloc %[[ST1]]
// CHECK: memref.dealloc %[[SCRATCH]]
// CHECK-NOT: hmx.matmul
// CHECK-NOT: memref.dma_start
// CHECK-NOT: arith.remui
func.func @pipeline(%a: memref<128x1024xf16>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %ca = memref.alloc() : memref<4x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c128 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf16>) outs(%ca : memref<4x32x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>)
  memref.dealloc %ca : memref<4x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}

// The same bridge before canonicalization, when the crouton array still travels
// as the pack loop's iter_arg: the same pipelined staged loop, and the whole
// array is retired just the same -- only the scratch survives.
// CHECK-LABEL: func.func @pipeline_carried
// CHECK: %[[STATE:.*]] = memref.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
// CHECK: hmx.bias_init %[[STATE]]
// CHECK-NOT: memref<4x32x16x32x2xf16, 1>
// CHECK: %[[WCRING:.*]] = memref.alloc() : memref<2x32x16x32x2xf16, 1>
// CHECK: hmx.pack_weight ins(%arg1, {{.*}}) outs(%[[WCRING]] :
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<4x2x16x32x2xf16, 1>
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// Two-slot ring; the prologue issues tile 0 through the parity slot select.
// CHECK: %[[SLOT0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// CHECK: %[[ST0:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[SLOT1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// CHECK: %[[ST1:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[SSEL0:.*]] = arith.select {{.*}}, %[[SLOT1]], %[[SLOT0]] : memref<32x1024xf16, 1>
// CHECK: %[[T0:.*]] = hmx.stage ins(%arg0, {{.*}} : memref<128x1024xf16>) outs(%[[SSEL0]], {{.*}} : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// The kernel carries the (token, slot) pair as iter_args; the mmas read the
// scratch at row 0 and the output tile is the unshifted induction variable.
// CHECK: scf.for %[[K:.*]] = {{.*}} iter_args(%[[T:.*]] = %[[T0]], %[[S:.*]] = %[[SSEL0]]) -> (i32, memref<32x1024xf16, 1>) {
// CHECK: %[[SSELK:.*]] = arith.select {{.*}}, %[[SLOT1]], %[[SLOT0]] : memref<32x1024xf16, 1>
// CHECK: %[[TNK:.*]] = hmx.stage ins(%arg0, {{.*}} : memref<128x1024xf16>) outs(%[[SSELK]], {{.*}} : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// CHECK: %[[ROW:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[S]] : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// CHECK: hmx.pack_act ins(%[[ROW]], %[[ZERO:.*]], %[[ZERO]] : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] : memref<1x32x16x32x2xf16, 1>) {count = 32 : i64}
// CHECK: hmx.acc_clear
// CHECK: hmx.mma %[[SCRATCH]], %[[WCRING]], %[[ZERO]], {{.*}}, {{.*}} {n_croutons = 1 : i32}
// CHECK: hmx.acc_read %[[STATE]], %[[ACC]], {{.*}}, {{.*}} {bias_set = 0 : i32}
// CHECK: scf.yield %[[TNK]], %[[SSELK]] : i32, memref<32x1024xf16, 1>
// Peeled epilogue: one await and compute, no stage.
// CHECK: hmx.await ins(%{{.*}} : i32) outs(%{{.*}} : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// CHECK-NOT: hmx.matmul
// CHECK-NOT: memref.dma_start
func.func @pipeline_carried(%a: memref<128x1024xf16>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %ca = memref.alloc() : memref<4x32x16x32x2xf16, 1>
  %pa = scf.for %i = %c0 to %c128 step %c1 iter_args(%carry = %ca) -> (memref<4x32x16x32x2xf16, 1>) {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf16>) outs(%carry : memref<4x32x16x32x2xf16, 1>)
    scf.yield %carry : memref<4x32x16x32x2xf16, 1>
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%pa, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>)
  return
}

// `auto` does not stage a shallow-K shape: Kt=4 is below the `kStageMinKTiles`
// floor (32), where the transfer is too small to hide the DMA engine's fixed
// cost behind the tile's compute. So the plain tile loop is emitted unchanged:
// the activation bridge and its whole crouton array are kept and the mmas read
// that array, exactly as for a shape the pipeline never applied to. There is no
// scratch and no staging ring, hence no `hmx.stage`/`hmx.await` anywhere. The
// shape here is Mt=2, Kt=4 (K=128): a `memref<64x128xf16>` source and a
// `2x4x16x32x2` activation crouton array.
// CHECK-LABEL: func.func @pipeline_shallow_kt
// CHECK: hmx.bias_init
// The activation array survives: the bridge fills it and the mmas read it.
// CHECK: %[[ACT:.*]] = memref.alloc() : memref<2x4x16x32x2xf16, 1>
// CHECK: scf.for {{.*}} {
// CHECK: hmx.pack_act ins(%arg0, {{.*}} : memref<64x128xf16>) outs(%[[ACT]] :
// The weight is the whole 2x4 crouton array, not a resident ring.
// CHECK: %[[W:.*]] = memref.alloc() : memref<2x4x16x32x2xf16, 1>
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<2x2x16x32x2xf16, 1>
// The plain (m, n) tile nest reads the activation array directly.
// CHECK: scf.for %[[M:.*]] = {{.*}} to
// CHECK: scf.for %[[N:.*]] = {{.*}} to
// CHECK: hmx.acc_clear
// CHECK: scf.for %[[K:.*]] = {{.*}} to
// CHECK: hmx.mma %[[ACT]], %[[W]], %[[M]], %[[N]], %[[K]] {n_croutons = 1 : i32}
// CHECK: hmx.acc_read {{.*}}, %[[ACC]], %[[M]], %[[N]] {bias_set = 0 : i32}
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: hmx.matmul
// CHECK-NOT: memref.dma_start
func.func @pipeline_shallow_kt(%a: memref<64x128xf16>, %w: memref<128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %ca = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c4 : index
    %cc = arith.remui %i, %c4 : index
    hmx.pack_act ins(%a, %r, %cc : memref<64x128xf16>) outs(%ca : memref<2x4x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<128x64xf16>) outs(%cw : memref<2x4x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>)
             outs(%ar : memref<2x2x16x32x2xf16, 1>)
  return
}
