//===- hexagon-unroll-and-jam.mlir ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -hexagon-unroll-and-jam='unroll-jam-factor=2' -split-input-file | FileCheck %s
// RUN: linalg-hexagon-opt %s -hexagon-unroll-and-jam='unroll-jam-factor=2 max-reduction-row-length=128' -split-input-file | FileCheck %s --check-prefix=CAP128

// A reduction row loop whose induction variable is i32 (as triton-shared
// emits it for rmsnorm). The pass must normalize the IV to index (Option 3A)
// and then jam the body by 2 -> two independent linalg.reduce chains. The
// tell-tale that normalization ran is the inserted `index to i32` cast (the
// source only has an `i32 to index` cast).
func.func @normalize_i32_iv(%in: memref<256xf16>, %out: memref<f32>,
                            %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.reduce ins(%in : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @normalize_i32_iv
// CHECK: arith.index_cast %{{.*}} : index to i32
// CHECK-COUNT-2: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// Same reduction loop but with an index-typed IV: the easy case. The pass must
// jam it just as well, and must NOT insert any index_cast (normalization is
// skipped for an already-index IV).
func.func @index_iv_not_normalized(%in: memref<256xf16>, %out: memref<f32>,
                                   %scratch: memref<32xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    memref.store %i, %scratch[%i] : memref<32xindex>
    linalg.reduce ins(%in : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @index_iv_not_normalized
// CHECK-NOT: arith.index_cast
// CHECK-COUNT-2: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// The core of this pass: the jammed copies must be INTERLEAVED round-robin
// (A1 B1 A2 B2 A3 B3), not appended (A1 A2 A3 B1 B2 B3). Interleaving is what
// puts each row's independent op next to the other row's dependent op so the
// in-order scheduler can overlap their latencies. The body here has three
// distinct positions (store, map, reduce); after the jam we must see the two
// copies of each position back-to-back, in position order.
func.func @interleave_order(%in: memref<256xf16>, %sq: memref<256xf16>,
                            %out: memref<f32>, %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.map { arith.mulf }
        ins(%in, %in : memref<256xf16>, memref<256xf16>)
        outs(%sq : memref<256xf16>)
    linalg.reduce ins(%sq : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// The step is doubled to 2 (two rows per iteration).
// CHECK-LABEL: func.func @interleave_order
// CHECK: scf.for %{{.*}} step %c2
// Position 0 (store) x2, then position 1 (map) x2, then position 2 (reduce) x2.
// CHECK:      memref.store
// CHECK-NEXT: memref.store
// CHECK:      linalg.map
// CHECK-NEXT: linalg.map
// CHECK:      linalg.reduce
// CHECK:      linalg.reduce
// CHECK-NOT:  linalg.reduce

// -----

// A loop carrying an accumulator via iter_args (so it has results) must be
// rejected: loopUnrollJamByFactor does not support loops with results, and the
// predicate gates on it. The single reduce must stay single.
func.func @has_iter_args(%in: memref<256xf16>, %out: memref<f32>) -> f32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (f32) : i32 {
    linalg.reduce ins(%in : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
    scf.yield %acc : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @has_iter_args
// CHECK: scf.for %{{.*}} iter_args
// CHECK-COUNT-1: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// Trip count 31 is not divisible by the factor 2 -> reject (no remainder loop
// is generated).
func.func @non_divisible(%in: memref<256xf16>, %out: memref<f32>,
                         %scratch: memref<31xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c31 = arith.constant 31 : i32
  scf.for %i = %c0 to %c31 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<31xi32>
    linalg.reduce ins(%in : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @non_divisible
// CHECK-COUNT-1: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// A purely elementwise loop (no reduction) is legal to jam but not profitable:
// there is no serial dependent chain to overlap, so the heuristic skips it.
func.func @elementwise_only(%a: memref<256xf16>, %b: memref<256xf16>,
                            %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                     affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<256xf16>) outs(%b : memref<256xf16>) {
      ^bb0(%in: f16, %out: f16):
        %x = arith.addf %in, %in : f16
        linalg.yield %x : f16
    }
  }
  return
}

// CHECK-LABEL: func.func @elementwise_only
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: linalg.generic

// -----

// A reduction whose row length C is within the default cap (1024) is jammed.
// 1024 is where the win begins to fade, so it is the last size we still treat.
func.func @row_length_at_cap(%in: memref<1024xf16>, %out: memref<f32>,
                             %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.reduce ins(%in : memref<1024xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @row_length_at_cap
// CHECK-COUNT-2: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// A reduction whose row length C exceeds the default cap (1024) is left
// un-jammed: past this the per-row elementwise stream dominates and jamming
// stops paying off.
func.func @row_length_over_cap(%in: memref<2048xf16>, %out: memref<f32>,
                               %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.reduce ins(%in : memref<2048xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @row_length_over_cap
// CHECK-COUNT-1: linalg.reduce
// CHECK-NOT: linalg.reduce

// -----

// A body containing a real inner scf.for is a loop *nest*, not a straight-line
// sequence, so it must be routed to upstream loopUnrollJamByFactor (which fuses
// the duplicated inner loops) rather than to our round-robin interleave. What
// reveals of the nest path: the outer step is doubled and an inner scf.for
// survives (upstream keeps the loop structure; it does not flatten the nest
// into an interleaved straight-line body the way our path would).
func.func @inner_loop_nest_uses_upstream(%in: memref<8xf16>, %out: memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      linalg.reduce ins(%in : memref<8xf16>) outs(%out : memref<f32>)
          dimensions = [0]
        (%v: f16, %init: f32) {
          %e = arith.extf %v : f16 to f32
          %a = arith.addf %init, %e : f32
          linalg.yield %a : f32
        }
    }
  }
  return
}

// CHECK-LABEL: func.func @inner_loop_nest_uses_upstream
// CHECK: scf.for %{{.*}} step %c2
// CHECK: scf.for

// -----

// The row-length cap is a live option, not a baked-in constant. This C=256
// reduction is jammed under the default cap (1024) but left un-jammed when the
// cap is lowered to 128 via max-reduction-row-length -- proving the option is
// actually threaded into the profitability heuristic.
func.func @row_length_cap_is_tunable(%in: memref<256xf16>, %out: memref<f32>,
                                     %scratch: memref<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    %idx = arith.index_cast %i : i32 to index
    memref.store %i, %scratch[%idx] : memref<32xi32>
    linalg.reduce ins(%in : memref<256xf16>) outs(%out : memref<f32>)
        dimensions = [0]
      (%v: f16, %init: f32) {
        %e = arith.extf %v : f16 to f32
        %a = arith.addf %init, %e : f32
        linalg.yield %a : f32
      }
  }
  return
}

// CHECK-LABEL: func.func @row_length_cap_is_tunable
// CHECK-COUNT-2: linalg.reduce
// CHECK-NOT: linalg.reduce

// CAP128-LABEL: func.func @row_length_cap_is_tunable
// CAP128-COUNT-1: linalg.reduce
// CAP128-NOT: linalg.reduce
