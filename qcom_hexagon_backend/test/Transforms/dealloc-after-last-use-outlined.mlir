//===- dealloc-after-last-use-outlined.mlir -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Synthetic input (hand-written, NOT produced by any tool).
//
// Reproduces the reported buffer-lifetime shape that crosses outlined calls:
//
//   %alloc_72 = memref.alloc() : memref<6x512x512xf32>          (large)
//     -> subview + vector.transfer_write   (a dead write, not the last use)
//     -> cast -> call @merged(...) which does `return %arg2`  (pass-through)
//        so the buffer flows out as the call RESULT %17
//     -> expand_shape %17 -> call @outlined_12 which returns FRESH allocs
//        (NOT a pass-through) => last use is that call.
//
// Expect a single memref.dealloc on %alloc_72 immediately AFTER the call to
// @outlined_12, and none before it.
//
// Byte size (chunk-bytes = 1048576):
//   memref<6x512x512xf32> = 6*512*512*4 = 6,291,456 B -> large.
//
// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:   -hexagon-dealloc-after-last-use='chunk-bytes=1048576' | FileCheck %s

// Pass-through callee: returns its third argument unchanged.
func.func private @merged(
    %a: memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>,
    %c: memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>)
    -> memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>> {
  return %c : memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>
}

// Terminal callee: returns fresh allocations, NOT any of its arguments.
func.func private @outlined_12(
    %in: memref<2x3x512x512xf32, strided<[?, ?, ?, ?], offset: ?>>)
    -> (memref<2x3x512x512xf32>, memref<2x3x512x512xf32>) {
  %a = memref.alloc() : memref<2x3x512x512xf32>
  %b = memref.alloc() : memref<2x3x512x512xf32>
  return %a, %b : memref<2x3x512x512xf32>, memref<2x3x512x512xf32>
}

func.func public @main(
    %c0arg: memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>,
    %c1arg: memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  %alloc_72 = memref.alloc() {alignment = 128 : i64} : memref<6x512x512xf32>

  // A dead write into a subview — a use, but not the last one.
  %sv = memref.subview %alloc_72[0, 0, 0] [1, 1, 32] [1, 1, 1]
      : memref<6x512x512xf32>
      to memref<32xf32, strided<[1], offset: 0>>
  memref.store %cst, %sv[%c0]
      : memref<32xf32, strided<[1], offset: 0>>

  // cast -> call @merged returns %arg2 (this buffer) -> %17.
  %cast_75 = memref.cast %alloc_72
      : memref<6x512x512xf32>
      to memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>
  %17 = call @merged(%c0arg, %c1arg, %cast_75)
      : (memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>,
         memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>,
         memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>)
      -> memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>

  // expand_shape the pass-through result, then feed the terminal callee.
  %expand_76 = memref.expand_shape %17 [[0, 1], [2], [3]]
      output_shape [2, 3, 512, 512]
      : memref<6x512x512xf32, strided<[?, ?, ?], offset: ?>>
      into memref<2x3x512x512xf32, strided<[?, ?, ?, ?], offset: ?>>
  %19:2 = call @outlined_12(%expand_76)
      : (memref<2x3x512x512xf32, strided<[?, ?, ?, ?], offset: ?>>)
      -> (memref<2x3x512x512xf32>, memref<2x3x512x512xf32>)
  return
}

// CHECK-LABEL: func.func public @main
// The alloc is preserved (this pass only inserts deallocs).
// CHECK:       %[[ALLOC:.+]] = memref.alloc() {alignment = 128 : i64} : memref<6x512x512xf32>
// No dealloc may appear before the terminal call...
// CHECK-NOT:   memref.dealloc
// CHECK:       call @outlined_12
// ...it must land right after the terminal call.
// CHECK:       memref.dealloc %[[ALLOC]] : memref<6x512x512xf32>

// -----

// Escape case: the buffer is returned by the entry function, so it must NEVER
// be deallocated.
func.func public @escapes() -> memref<6x512x512xf32> {
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<6x512x512xf32>
  return %alloc : memref<6x512x512xf32>
}

// CHECK-LABEL: func.func public @escapes
// CHECK-NOT:   memref.dealloc

// -----

// Hexagon-memory VTCM allocation without a pre-existing dealloc. The buffer
// must remain live through the DMA wait and receive a dialect-native free.
func.func public @hexmem_dma_missing_dealloc(
    %src: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  memref.dma_start %src[%c0], %dst[%c0], %c32, %tag[%c0]
      : memref<32xf32>, memref<32xf32, 1>, memref<1xi32>
  memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
  return
}

// CHECK-LABEL: func.func public @hexmem_dma_missing_dealloc
// CHECK:       memref.dma_start
// CHECK:       memref.dma_wait
// CHECK:       hexagonmem.dealloc %{{.*}} : memref<32xf32, 1>

// -----

// An existing late dealloc is moved to the last completed use.
func.func public @hexmem_dma_move_dealloc(
    %src: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  memref.dma_start %src[%c0], %dst[%c0], %c32, %tag[%c0]
      : memref<32xf32>, memref<32xf32, 1>, memref<1xi32>
  memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
  hexagonmem.dealloc %dst : memref<32xf32, 1>
  %after = arith.constant 7 : index
  return
}

// CHECK-LABEL: func.func public @hexmem_dma_move_dealloc
// CHECK:       memref.dma_wait
// CHECK:       hexagonmem.dealloc %{{.*}} : memref<32xf32, 1>
// CHECK:       %{{.*}} = arith.constant 7 : index

// -----

// A VTCM buffer used by a DMA inside a loop is freed after the loop, not after
// the first iteration.
func.func public @hexmem_dma_loop_lifetime(
    %src: memref<2x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  scf.for %i = %c0 to %c2 step %c1 {
    memref.dma_start %src[%i, %c0], %dst[%c0], %c32, %tag[%c0]
        : memref<2x32xf32>, memref<32xf32, 1>, memref<1xi32>
    memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
  }
  return
}

// CHECK-LABEL: func.func public @hexmem_dma_loop_lifetime
// CHECK:       scf.for
// CHECK:       }
// CHECK:       hexagonmem.dealloc %{{.*}} : memref<32xf32, 1>

// -----

// A DMA start with no lexically provable matching wait is retained. This is
// the conservative behavior needed for a loop-carried or otherwise
// asynchronous transfer whose completion cannot be established locally.
func.func public @hexmem_dma_unknown_completion(
    %src: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  scf.for %i = %c0 to %c2 step %c1 {
    memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
    memref.dma_start %src[%c0], %dst[%c0], %c32, %tag[%c0]
        : memref<32xf32>, memref<32xf32, 1>, memref<1xi32>
  }
  return
}

// CHECK-LABEL: func.func public @hexmem_dma_unknown_completion
// CHECK-NOT:   hexagonmem.dealloc

// Double-buffer prefetches are waited in the following loop iteration. The
// explicit db_prefetch/db_generic protocol proves that the final prefetch is
// suppressed, so the buffer can be released after the loop.
func.func public @hexmem_dma_loop_carried_prefetch(
    %src: memref<2x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloca() : memref<1xi32>
  scf.for %i = %c0 to %c2 step %c1 {
    %condition = arith.cmpi ne, %i, %c0 : index
    scf.if %condition {
      memref.dma_start %src[%i, %c0], %dst[%c0], %c32, %tag[%c0]
          : memref<2x32xf32>, memref<32xf32, 1>, memref<1xi32>
    } {db_prefetch}
    memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
    %value = memref.load %dst[%c0] : memref<32xf32, 1>
    memref.store %value, %dst[%c0] : memref<32xf32, 1>
  } {db_generic = 0 : i64}
  return
}

// CHECK-LABEL: func.func public @hexmem_dma_loop_carried_prefetch
// CHECK:       scf.for
// CHECK:       } {db_generic = 0 : i64}
// CHECK:       hexagonmem.dealloc %{{.*}} : memref<32xf32, 1>

// A tagged tiled loop is lowered as parallel work even though its structured
// representation is scf.for. Do not move a VTCM free based only on lexical
// last-use inside this region.
func.func public @hexmem_all_parallel_lifetime(
    %src: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  scf.for %i = %c0 to %c2 step %c1 {
    memref.dma_start %src[%c0], %dst[%c0], %c32, %tag[%c0]
        : memref<32xf32>, memref<32xf32, 1>, memref<1xi32>
    memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
  } {all_parallel}
  hexagonmem.dealloc %dst : memref<32xf32, 1>
  %after = arith.constant 7 : index
  return
}

// CHECK-LABEL: func.func public @hexmem_all_parallel_lifetime
// CHECK:       scf.for
// CHECK:       } {all_parallel}
// CHECK:       hexagonmem.dealloc
// CHECK:       %{{.*}} = arith.constant 7 : index

// -----

// An outer VTCM allocation with only all_parallel users still receives a free
// after the parallel loop once all workers have joined.
func.func public @hexmem_all_parallel_missing_dealloc(
    %src: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %dst = hexagonmem.alloc() : memref<32xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  scf.for %i = %c0 to %c2 step %c1 {
    memref.dma_start %src[%c0], %dst[%c0], %c32, %tag[%c0]
        : memref<32xf32>, memref<32xf32, 1>, memref<1xi32>
    memref.dma_wait %tag[%c0], %c32 : memref<1xi32>
  } {all_parallel}
  return
}

// CHECK-LABEL: func.func public @hexmem_all_parallel_missing_dealloc
// CHECK:       } {all_parallel}
// CHECK:       hexagonmem.dealloc %{{.*}} : memref<32xf32, 1>
