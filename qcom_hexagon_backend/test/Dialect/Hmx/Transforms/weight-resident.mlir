//===- weight-resident.mlir - constant weights become resident VTCM -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The constant-weight anchor of matmul-to-hmx, one stage later: after
// bufferization the prepacked constant is a `memref.get_global`, which the HMX
// engine cannot read (its weight has to be in VTCM). This pass gives it a VTCM
// buffer, records where its contents come from and declares the residency on
// the module, so the budget readers and the runtime see one number.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// One 64x64 f16 weight is a 2x2 grid of 32x32 croutons: 2*2*16*32*2 elements *
// 2 bytes = 8192 bytes. The pass turns the DDR global into a VTCM buffer whose
// contents the runtime copies in once, and declares the footprint on the module
// for the static budget readers.
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // The prepacked constant has to stay addressable so the lowering can fill the
  // resident buffer from it; the pass makes it public so symbol DCE keeps it.
  // CHECK: memref.global "public" constant @__weight
  memref.global "private" constant @__weight : memref<2x2x16x32x2xf16> = dense<1.000000e+00> {alignment = 128 : i64}

  // CHECK-LABEL: func.func @constant_weight
  func.func @constant_weight() {
    // The buffer is not filled here: its lowering calls the runtime, which
    // copies the constant in once and pins it.
    // CHECK: %[[W:.*]] = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @__weight}} : memref<2x2x16x32x2xf16, 1>
    // CHECK: hmx.matmul ins({{.*}}, %[[W]] : memref<2x2x16x32x2xf16, 1>
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %w = memref.get_global @__weight : memref<2x2x16x32x2xf16>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16>)
               outs(%r : memref<2x2x16x32x2xf16, 1>)
    return
  }
}

// -----

// A non-constant weight is already a VTCM buffer and must be left exactly as it
// is: the residency path only claims the compile-time constant case.
module {
  // CHECK-LABEL: func.func @runtime_weight_untouched
  // CHECK-NOT: hmx.weight_resident
  // CHECK-NOT: hexagonmem.alloc
  // CHECK: hmx.matmul
  func.func @runtime_weight_untouched() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
               outs(%r : memref<2x2x16x32x2xf16, 1>)
    return
  }
}

// -----

// A runtime weight that still carries its pack bridge is left alone unless the
// pre-pack contract is requested: with the option off the kernel keeps packing
// it every launch, and nothing is declared resident.
module {
  // CHECK-LABEL: func.func @runtime_weight_pack_untouched
  // CHECK: hmx.pack_weight
  // CHECK-NOT: hmx.weight_resident
  // CHECK-NOT: hmx.weight_prepack
  // CHECK-NOT: hmx.weight_resident_bytes
  func.func @runtime_weight_pack_untouched(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %aa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    return
  }
}
