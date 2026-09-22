//===- weight-resident-runtime.mlir - runtime weights become resident ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// P2: a runtime weight (a function argument) is bridged by a `hmx.pack_weight`
// loop, paid on every launch. With `prepack-runtime-weights` on, the pass
// replaces that bridge with a resident VTCM buffer whose bytes the host
// pre-packs, records the footprint on the module, and publishes the slot/layout
// contract for the host (`hmx.weight_prepack`).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// The host pre-pack contract names the function, the slot, the logical shape and
// the crouton shape (MLIR prints `"` inside the string as `\22`). Attributes
// print alphabetically: prepack, prepack_layout, resident_bytes.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_prepack_layout = "{{[{]\\22ndims\\22:5,\\22results\\22:\[\[\[1,32\],\[2,2\],\[4,1\]\],\[\[0,32\],\[3,1\]\]\][}]}}"
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // CHECK-LABEL: func.func @runtime_weight
  func.func @runtime_weight(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    // The bridge: one pack_weight per 32x32 tile, from the runtime argument.
    // CHECK-NOT: hmx.pack_weight
    // CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
    // CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %aa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    // The engine reads the resident buffer directly; the bridge and its array
    // are gone.
    // CHECK: hmx.matmul ins({{.*}}, %[[W]] : memref<2x2x16x32x2xf16, 1>
    // CHECK-NOT: memref.dealloc %[[W]]
    // CHECK: return
    hmx.matmul ins(%aa, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %aa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

module {
// Real kernels wrap the pack source in a layout-only `reinterpret_cast` with
// a dense-equivalent strided layout (bufferization of the tensor slice chain).
// The pass must see through it to the argument; a bare dyn_cast would silently
// keep the per-launch bridge (this exact shape shipped and ran pack_weight on
// device while the gate was on).
// CHECK-LABEL: func.func @runtime_weight_viewed
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2
// CHECK: hmx.matmul ins({{.*}}, %[[W]] : memref<2x2x16x32x2xf16, 1>
// CHECK-NOT: hmx.pack_weight
func.func @runtime_weight_viewed(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %view = memref.reinterpret_cast %w to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
  %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>) outs(%ca : memref<2x2x16x32x2xf16, 1>)
  }
  %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%view, %r, %cc : memref<64x64xf16, strided<[64, 1]>>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
  memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
  return
  }
}

// -----

// A real Triton kernel receives the weight as an *unranked* memref
// (`memref<*xf16>`): the entry argument has no rank, and the kernel asserts the
// layout with a static rank/shape `reinterpret_cast`. With offset 0 and dense
// row-major strides the view provably covers the whole argument, so the
// argument is the pre-pack key and the view's static shape is the contract
// shape. Before the fix the `dyn_cast<MemRefType>` on the unranked source
// returned null and the case was skipped entirely, so the per-launch bridge
// survived on real kernels even with the option on.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_unranked\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_resident_bytes = 8192 : i64
// CHECK-LABEL: func.func @runtime_weight_unranked
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
// CHECK: hmx.matmul ins({{.*}}, %[[W]] : memref<2x2x16x32x2xf16, 1>
// CHECK-NOT: hmx.pack_weight
module {
  func.func @runtime_weight_unranked(%a: memref<64x64xf16>, %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = memref.reinterpret_cast %w to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xf16> to memref<64x64xf16, strided<[64, 1]>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>) outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x64xf16, strided<[64, 1]>>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// An unranked source alone does not make the argument the weight: a *dynamic*
// view offset (`offset: [%n]`, the program-id N offset a real decode kernel
// uses) means the argument's bytes are not the weight's bytes. Pre-packing the
// whole argument there would silently compute on the wrong data, so the pass
// must keep the per-launch `hmx.pack_weight` bridge and declare no residency.
// This is a known P2 limitation, deliberately not guessed around here.
// The module footprint must stay untouched too: `hmx.weight_resident_bytes`
// prints before the function, so it needs its own CHECK-NOT ahead of the label.
// CHECK-NOT: hmx.weight_resident
// CHECK-LABEL: func.func @runtime_weight_dynamic_offset
// CHECK-NOT: hmx.weight_resident
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK-NOT: hmx.weight_resident
module {
  func.func @runtime_weight_dynamic_offset(%a: memref<64x64xf16>, %w: memref<*xf16>, %off: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = memref.reinterpret_cast %w to offset: [%off], sizes: [64, 64], strides: [64, 1] : memref<*xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x64xf16>) outs(%ca : memref<2x2x16x32x2xf16, 1>)
    }
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %cc = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%view, %r, %cc : memref<64x64xf16, strided<[64, 1], offset: ?>>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}
