//===- weight-resident-nsplit.mlir - whole-weight residency (B2) ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A decode kernel splits N across programs: each program packs one N block of a
// wider weight, so the pack source is
//
//   reinterpret_cast %w to offset: [n0], sizes: [K, BN], strides: [N, 1]
//
// The view's row stride is the *whole* N, so the argument really is the whole
// `[K, N]` weight and one resident copy of it serves every program. The pass
// rebases residency to the whole weight (contract `shape = [K, N]`, `crouton =
// [N/32, K/32, 16, 32, 2]`, whole byte count) and feeds each `hmx.matmul` a
// `memref.subview` over the crouton grid's N tiles. Anything the pass cannot pin
// -- an offset that is not provably tile-aligned, or a view that is not a plain
// N block -- keeps the old per-launch bridge, never a guessed resident.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// [1] dynamic, tile-aligned N offset: whole-weight residency + subview.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_nsplit\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_prepack_layout = "{{[{]\\22ndims\\22:5,\\22results\\22:\[\[\[1,32\],\[2,2\],\[4,1\]\],\[\[0,32\],\[3,1\]\]\][}]}}"
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // CHECK-LABEL: func.func @runtime_weight_nsplit
  // CHECK-NOT: hmx.pack_weight
  func.func @runtime_weight_nsplit(%a: memref<64x64xf16>, %w: memref<*xf16>,
                                   %pid: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %n0 = arith.muli %pid, %c32 : index
    // One N block of the whole [64, 64]: row stride 64, offset pid * 32.
    %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
        strides: [64, 1]
        : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>)
          outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x32xf16, strided<[64, 1], offset: ?>>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    // The resident holds the whole [N, K] = [64, 64] (crouton [2, 2, 16, 32, 2],
    // 8192 B); the matmul reads the subview for this program's N block (the N
    // tile offset lands on dim 0, and dim 0's stride is 2 croutons = 2048 elems).
    // CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
    // CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
    // CHECK: %[[N0C:.*]] = arith.divui
    // CHECK: %[[SUB:.*]] = memref.subview %[[W]][%[[N0C]], 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<2x2x16x32x2xf16, 1> to memref<1x2x16x32x2xf16, strided<[2048, 1024, 64, 2, 1], offset: ?>, 1>
    // CHECK: hmx.matmul ins({{.*}}, %[[SUB]] : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, strided<[2048, 1024, 64, 2, 1], offset: ?>, 1>)
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [2] offset not provably tile-aligned: keep the old per-launch bridge.
// `pid * 16` cannot be shown to land on a 32-element crouton edge, so the whole
// weight cannot be pinned; pre-packing it would silently compute on the wrong
// bytes. No residency is declared anywhere (the module footprint prints before
// the function, so it needs its own CHECK-NOT ahead of the label).
// CHECK-NOT: hmx.weight_resident
// CHECK-LABEL: func.func @runtime_weight_nsplit_unaligned
// CHECK-NOT: hmx.weight_resident
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK-NOT: hmx.weight_resident
module {
  func.func @runtime_weight_nsplit_unaligned(%a: memref<64x64xf16>,
                                              %w: memref<*xf16>, %pid: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %n0 = arith.muli %pid, %c16 : index
    %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
        strides: [64, 1]
        : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>)
          outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x32xf16, strided<[64, 1], offset: ?>>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [3] dense whole view, static 0 offset: the old path does not regress.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_whole\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // CHECK-LABEL: func.func @runtime_weight_whole
  // CHECK-NOT: hmx.pack_weight
  func.func @runtime_weight_whole(%a: memref<64x64xf16>, %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = memref.reinterpret_cast %w to offset: [0], sizes: [64, 64],
        strides: [64, 1]
        : memref<*xf16> to memref<64x64xf16, strided<[64, 1]>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>)
          outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x64xf16, strided<[64, 1]>>)
          outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    // The whole argument is the weight: the matmul reads the resident itself, no
    // subview.
    // CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
    // CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
    // CHECK-NOT: memref.subview
    // CHECK: hmx.matmul ins({{.*}}, %[[W]] : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// [4] static tile-aligned N offset: same B2 shape, folded crouton offset.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_static\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // CHECK-LABEL: func.func @runtime_weight_static
  // CHECK-NOT: hmx.pack_weight
  func.func @runtime_weight_static(%a: memref<64x64xf16>, %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = memref.reinterpret_cast %w to offset: [32], sizes: [64, 32],
        strides: [64, 1]
        : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: 32>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>)
          outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x32xf16, strided<[64, 1], offset: 32>>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    // CHECK: %[[W:.*]] = hexagonmem.alloc({{.*}}) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
    // Static offset 32 -> crouton offset 1, folded to a constant.
    // CHECK: memref.subview %[[W]][{{.*}}, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1]
    // CHECK: hmx.matmul
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [5] A view as wide as the whole N is not an N block of a wider weight: it is
// the loop's K block of an activation consumed as a weight (the flash-attention
// shape, where the row stride happens to equal the block width). Making it
// resident would hold one block while the loop's offset walked past the
// resident's end, so the bridge must stay.
module {
  // CHECK-LABEL: func.func @runtime_kblock_whole_n
  // CHECK-NOT: hmx.weight_prepack
  // CHECK-NOT: hexagonmem.alloc
  // CHECK-NOT: memref.subview
  // CHECK: hmx.pack_weight
  // CHECK: hmx.matmul
  func.func @runtime_kblock_whole_n(%a: memref<256x64xf16>, %w: memref<*xf16>,
                                    %k0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %off = arith.muli %k0, %c64 : index
    // The whole N is 64 and so is the view: the offset selects rows, not columns.
    %view = memref.reinterpret_cast %w to offset: [%off], sizes: [64, 64],
        strides: [64, 1]
        : memref<*xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %ca = memref.alloc() : memref<8x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c8 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<256x64xf16>)
          outs(%ca : memref<8x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x64xf16, strided<[64, 1], offset: ?>>)
          outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<8x2x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<8x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
        outs(%ar : memref<8x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<8x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<8x2x16x32x2xf16, 1>
    return
  }
}

// -----

// [6] An offset that is provably a multiple of the whole N is a row (K) offset,
// even when the view is narrower than N: in a row-major [K, N] matrix every whole
// number of rows is a multiple of N, and the N-block model only admits column
// offsets. `%k0 * 128` over a row stride of 128 is such a row offset, so the
// bridge stays.
module {
  // CHECK-LABEL: func.func @runtime_kblock_wider_n
  // CHECK-NOT: hmx.weight_prepack
  // CHECK-NOT: hexagonmem.alloc
  // CHECK-NOT: memref.subview
  // CHECK: hmx.pack_weight
  // CHECK: hmx.matmul
  func.func @runtime_kblock_wider_n(%a: memref<64x64xf16>, %w: memref<*xf16>,
                                    %k0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %off = arith.muli %k0, %c128 : index
    // Row stride 128 > block width 32, but the offset is a whole number of rows.
    %view = memref.reinterpret_cast %w to offset: [%off], sizes: [64, 32],
        strides: [128, 1]
        : memref<*xf16> to memref<64x32xf16, strided<[128, 1], offset: ?>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>)
          outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x32xf16, strided<[128, 1], offset: ?>>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}
