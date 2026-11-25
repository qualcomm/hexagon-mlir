//===- vtcm_mask_widening.mlir --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:     -pass-pipeline='builtin.module(func.func(vtcm-mask-widening))' \
// RUN: | FileCheck %s
//
// Tests verifying that VTCMMaskWideningPass retypes i1 tensor operands of
// any op VTCMTilingPass would select for VTCM staging to i8 in place, before
// VTCMTilingPass ever runs -- covering both the small/fully-prefetched case
// and the large/tiled-and-sliced case, since both of VTCMTilingPass's own
// staging paths call the same VTCM-copy helpers regardless of size.

// -----

// Small tensor that fits entirely in VTCM (prefetch path): an i1 `ins`
// operand is widened to i8 via a widening linalg.generic, and the original
// generic's block argument gets a transient trunci back to i1.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_ins_small(%mask: tensor<128x256xi1>, %x: tensor<128x256xf32>, %z: tensor<128x256xf32>) -> tensor<128x256xf32> {
   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%mask, %x : tensor<128x256xi1>, tensor<128x256xf32>)
           outs(%z : tensor<128x256xf32>) {
   ^bb0(%m: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %m, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<128x256xf32>
   return %t3 : tensor<128x256xf32>
 }
}

// CHECK-LABEL: @i1_ins_small
// CHECK-SAME: %[[MASK:.+]]: tensor<128x256xi1>
//
// The i1 mask is cast to i8 via a widening linalg.generic before the
// original op.
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<128x256xi8>
// CHECK: %[[WIDENED:.+]] = linalg.generic {{.*}} ins(%[[MASK]] : tensor<128x256xi1>) outs(%[[EMPTY]] : tensor<128x256xi8>)
// CHECK: %[[EXT:.+]] = arith.extui %{{.*}} : i1 to i8
// CHECK: linalg.yield %[[EXT]] : i8
//
// The original generic now takes the i8 tensor and truncs it back to i1
// inside its own body.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[WIDENED]]
// CHECK: ^bb0(%[[ARG:.+]]: i8,
// CHECK: %[[TRUNC:.+]] = arith.trunci %[[ARG]] : i8 to i1
// CHECK: arith.select %[[TRUNC]]

// -----

// Large tensor that exceeds VTCM budget (tiled/sliced path): the i1 `ins`
// operand still gets widened, even though VTCMTilingPass will later stage it
// via per-tile copies rather than a single upfront prefetch.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_ins_large(%mask: tensor<4096x8192xi1>, %x: tensor<4096x8192xf32>, %z: tensor<4096x8192xf32>) -> tensor<4096x8192xf32> {
   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%mask, %x : tensor<4096x8192xi1>, tensor<4096x8192xf32>)
           outs(%z : tensor<4096x8192xf32>) {
   ^bb0(%m: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %m, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<4096x8192xf32>
   return %t3 : tensor<4096x8192xf32>
 }
}

// CHECK-LABEL: @i1_ins_large
// CHECK-SAME: %[[MASK:.+]]: tensor<4096x8192xi1>
//
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4096x8192xi8>
// CHECK: %[[WIDENED:.+]] = linalg.generic {{.*}} ins(%[[MASK]] : tensor<4096x8192xi1>) outs(%[[EMPTY]] : tensor<4096x8192xi8>)
// CHECK: arith.extui %{{.*}} : i1 to i8
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[WIDENED]]
// CHECK: arith.trunci %{{.*}} : i8 to i1

// -----

// Write-only i1 `outs` operand: a fresh i8 tensor.empty replaces the i1
// init, and the generic's own declared result type is updated to i8 in
// place (no RAUW of the whole op).

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_outs_write_only(%x: tensor<128x256xf32>, %y: tensor<128x256xf32>, %m: tensor<128x256xi1>, %z: tensor<128x256xf32>) -> tensor<128x256xf32> {
   %mask = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%x, %y : tensor<128x256xf32>, tensor<128x256xf32>)
           outs(%m : tensor<128x256xi1>) {
   ^bb0(%a: f32, %b: f32, %out: i1):
     %cmp = arith.cmpf ogt, %a, %b : f32
     linalg.yield %cmp : i1
   } -> tensor<128x256xi1>

   %sel = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%mask, %x : tensor<128x256xi1>, tensor<128x256xf32>)
           outs(%z : tensor<128x256xf32>) {
   ^bb0(%mm: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %mm, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<128x256xf32>
   return %sel : tensor<128x256xf32>
 }
}

// CHECK-LABEL: @i1_outs_write_only
//
// The write-only i1 init is replaced by a fresh i8 tensor.empty -- no cast,
// no copy of the original i1 init.
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<128x256xi8>
// CHECK: %[[MASK:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<128x256xi8>)
// CHECK: ^bb0(%{{.+}}: f32, %{{.+}}: f32, %[[OUT:.+]]: i8):
// CHECK: %[[CMP:.+]] = arith.cmpf
// CHECK: %[[EXT:.+]] = arith.extui %[[CMP]] : i1 to i8
// CHECK: linalg.yield %[[EXT]] : i8
//
// The mask's own declared result type is i8 -- the downstream consumer
// takes it directly with no intervening cast op, truncing back to i1
// inside its own body instead.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[MASK]]
// CHECK: arith.trunci %{{.*}} : i8 to i1

// -----

// A generic nested inside a scf.for annotated `tiled_generic` (already
// tiled by an earlier pass) is skipped entirely, mirroring VTCMTilingPass's
// own skip so both passes agree on which ops are even considered.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @skip_already_tiled(%mask: tensor<128x256xi1>, %x: tensor<128x256xf32>, %z: tensor<128x256xf32>) -> tensor<128x256xf32> {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c128 = arith.constant 128 : index
   %r = scf.for %i = %c0 to %c128 step %c1 iter_args(%acc = %z) -> tensor<128x256xf32> {
     %t3 = linalg.generic {
             indexing_maps = [#map, #map, #map],
             iterator_types = ["parallel", "parallel"]}
             ins(%mask, %x : tensor<128x256xi1>, tensor<128x256xf32>)
             outs(%acc : tensor<128x256xf32>) {
     ^bb0(%m: i1, %in: f32, %out: f32):
       %sel = arith.select %m, %in, %out : f32
       linalg.yield %sel : f32
     } -> tensor<128x256xf32>
     scf.yield %t3 : tensor<128x256xf32>
   } {tiled_generic}
   return %r : tensor<128x256xf32>
 }
}

// CHECK-LABEL: @skip_already_tiled
// The mask operand is left completely untouched -- no widening cast, no
// tensor.empty of i8, no trunci in the body.
// CHECK-NOT: tensor<128x256xi8>
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.trunci
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<128x256xi1>, tensor<128x256xf32>)

// -----

// A bare tensor.extract reading directly from a write-only i1 outs result:
// the producing generic's result type is retyped to i8 in place via setType
// (no RAUW), so the tensor.extract's operand silently becomes i8-typed too,
// leaving its own declared i1 result type stale. The separate
// tensor.extract sweep patches it: retype the extract's result to i8, and
// insert a trunci immediately after so its original (i1) uses keep working.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @tensor_extract_sweep(%x: tensor<128x256xf32>, %y: tensor<128x256xf32>, %m: tensor<128x256xi1>, %idx0: index, %idx1: index) -> i1 {
   %mask = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%x, %y : tensor<128x256xf32>, tensor<128x256xf32>)
           outs(%m : tensor<128x256xi1>) {
   ^bb0(%a: f32, %b: f32, %out: i1):
     %cmp = arith.cmpf ogt, %a, %b : f32
     linalg.yield %cmp : i1
   } -> tensor<128x256xi1>

   %extracted = tensor.extract %mask[%idx0, %idx1] : tensor<128x256xi1>
   return %extracted : i1
 }
}

// CHECK-LABEL: @tensor_extract_sweep
// CHECK: %[[MASK:.+]] = linalg.generic
// CHECK-SAME: outs(%{{.+}} : tensor<128x256xi8>)
// CHECK: %[[EXTRACTED:.+]] = tensor.extract %[[MASK]][%{{.*}}, %{{.*}}] : tensor<128x256xi8>
// CHECK: %[[TRUNC:.+]] = arith.trunci %[[EXTRACTED]] : i8 to i1
// CHECK: return %[[TRUNC]] : i1

// -----

// A write-only i1 outs generic's result flows through a tensor.expand_shape
// before reaching its VTCM-tiling-selected consumer (the shape this pass
// originally missed: native_layer_norm's row-validity mask is computed rank-1
// then tensor.expand_shape'd to rank-2 before being consumed). Retyping the
// mask generic's result to i8 leaves the expand_shape's own declared result
// type stale (still i1) relative to its now-i8 operand, and the verifier
// rejects the mismatch unless the expand_shape's result is retyped too, in
// the same walk and before its downstream consumer is reached.

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_through_expand_shape(%a: tensor<256xf32>, %b: tensor<256xf32>, %x: tensor<1x256xf32>, %z: tensor<1x256xf32>) -> tensor<1x256xf32> {
   %m = tensor.empty() : tensor<256xi1>
   %mask = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel"]}
           ins(%a, %b : tensor<256xf32>, tensor<256xf32>)
           outs(%m : tensor<256xi1>) {
   ^bb0(%av: f32, %bv: f32, %out: i1):
     %cmp = arith.cmpf ogt, %av, %bv : f32
     linalg.yield %cmp : i1
   } -> tensor<256xi1>

   %expanded = tensor.expand_shape %mask [[0, 1]] output_shape [1, 256]
       : tensor<256xi1> into tensor<1x256xi1>

   %sel = linalg.generic {
           indexing_maps = [#map1, #map1, #map1],
           iterator_types = ["parallel", "parallel"]}
           ins(%expanded, %x : tensor<1x256xi1>, tensor<1x256xf32>)
           outs(%z : tensor<1x256xf32>) {
   ^bb0(%mm: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %mm, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<1x256xf32>
   return %sel : tensor<1x256xf32>
 }
}

// CHECK-LABEL: @i1_through_expand_shape
//
// The write-only i1 mask generic's result is retyped to i8 in place.
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<256xi8>
// CHECK: %[[MASK:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<256xi8>)
// CHECK: %[[EXT:.+]] = arith.extui %{{.*}} : i1 to i8
// CHECK: linalg.yield %[[EXT]] : i8
//
// The expand_shape's own declared result type is retyped to i8 to match its
// now-widened source -- no separate cast op, no RAUW, just the reshape's
// result type corrected in place.
// CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[MASK]] {{.*}} : tensor<256xi8> into tensor<1x256xi8>
//
// The downstream consumer takes the i8-typed reshape result directly and
// truncs back to i1 inside its own body, exactly as if it had always
// consumed an i8 value.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[EXPANDED]]
// CHECK: %[[TRUNC:.+]] = arith.trunci %{{.*}} : i8 to i1
// CHECK: arith.select %[[TRUNC]]

// -----

// Symmetric with @i1_through_expand_shape above but exercises
// tensor.collapse_shape instead of tensor.expand_shape.

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @i1_through_collapse_shape(%a: tensor<1x256xf32>, %b: tensor<1x256xf32>, %x: tensor<256xf32>, %z: tensor<256xf32>) -> tensor<256xf32> {
   %m = tensor.empty() : tensor<1x256xi1>
   %mask = linalg.generic {
           indexing_maps = [#map1, #map1, #map1],
           iterator_types = ["parallel", "parallel"]}
           ins(%a, %b : tensor<1x256xf32>, tensor<1x256xf32>)
           outs(%m : tensor<1x256xi1>) {
   ^bb0(%av: f32, %bv: f32, %out: i1):
     %cmp = arith.cmpf ogt, %av, %bv : f32
     linalg.yield %cmp : i1
   } -> tensor<1x256xi1>

   %collapsed = tensor.collapse_shape %mask [[0, 1]]
       : tensor<1x256xi1> into tensor<256xi1>

   %sel = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel"]}
           ins(%collapsed, %x : tensor<256xi1>, tensor<256xf32>)
           outs(%z : tensor<256xf32>) {
   ^bb0(%mm: i1, %in: f32, %out: f32):
     %c0 = arith.constant 0.0 : f32
     %s = arith.select %mm, %in, %c0 : f32
     linalg.yield %s : f32
   } -> tensor<256xf32>
   return %sel : tensor<256xf32>
 }
}

// CHECK-LABEL: @i1_through_collapse_shape
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x256xi8>
// CHECK: %[[MASK:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1x256xi8>)
// CHECK: arith.extui %{{.*}} : i1 to i8
// CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MASK]] {{.*}} : tensor<1x256xi8> into tensor<256xi8>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[COLLAPSED]]
// CHECK: arith.trunci %{{.*}} : i8 to i1
