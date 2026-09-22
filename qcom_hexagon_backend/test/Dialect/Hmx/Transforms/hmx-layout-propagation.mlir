//===- layout-propagation.mlir - elementwise chains stay in crouton ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// M3 (S1 "layout native") acceptance, at the IR level.
//
// The two `hmx.matmul`s of a linear-attention score chain share one crouton
// layout only across *cheap pure-f16* segments: an f16 scale between them
// commutes with the crouton permutation and runs *on the crouton array*
// (`@cheap_scale_chain`, `@map_between` in matmul-to-hmx.mlir).
//
// A mask `where` between them stays row-major (`@mask_dot_arg`,
// `@mask_dot_chain`): moving `select`/`cmpi`/index work into crouton order
// costs ~10x the same work in row-major (LWP, 2026-09-18), regressing naive
// linear attention 2.56x, while the removed round trip is <1% of runtime.
// The negative case is the co-scheduling boundary: a *reduction* spans
// croutons, so one conversion must survive in front of it. Folding through it
// would be over-folding.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
// Narrowed 1: a `where` mask stays row-major.
//
// `mask` is a full-size `i1` operand that cannot carry a crouton encoding (the
// encoding is f16-only), and `select` is not cheap f16 arithmetic, so the fold
// refuses: the first read-out is unpacked, the mask runs row-major, and the
// masked result is repacked for the second matmul.
//----------------------------------------------------------------------------//
// CHECK-LABEL: func.func @mask_dot_arg
// CHECK: hmx.matmul
// The first read-out comes back to row-major ...
// CHECK: hmx.unpack_acc
// ... the mask runs there ...
// CHECK: linalg.generic {{.*}} tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: } -> tensor<64x64xf16>
// ... and the masked result is repacked for the second matmul.
// CHECK: hmx.pack_act
// CHECK: hmx.matmul
// Only the final read-out reaches row-major.
// CHECK: hmx.unpack_acc
func.func @mask_dot_arg(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                        %c: tensor<64x64xf16>, %mask: tensor<64x64xi1>)
    -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %masked = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mask, %m0 : tensor<64x64xi1>, tensor<64x64xf16>) outs(%e1 : tensor<64x64xf16>) {
  ^bb0(%m: i1, %x: f16, %o: f16):
    %z = arith.constant 0.000000e+00 : f16
    %s = arith.select %m, %x, %z : f16
    linalg.yield %s : f16
  } -> tensor<64x64xf16>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %z2 = arith.constant 0.000000e+00 : f16
  %i2 = linalg.fill ins(%z2 : f16) outs(%e2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%masked, %c : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%i2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}

//----------------------------------------------------------------------------//
// Narrowed 2: the natural mask *chain* stays row-major.
//
// The mask is index arithmetic, materialised the way `tl.arange` produces it:
// `arange -> broadcast(i) -> broadcast(j) -> cmpi sge -> select`. Every link
// of it is outside the cheap-f16 rule (`linalg.index`, i32, `cmpi`,
// `select`), so the whole chain stays row-major: the `select` cannot move
// onto the crouton array even though its f16 operand is the read-out.
//----------------------------------------------------------------------------//
// CHECK-LABEL: func.func @mask_dot_chain
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
// The select runs row-major on the unpacked read-out ...
// CHECK: linalg.generic {{.*}} tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: } -> tensor<64x64xf16>
// ... and its result is repacked.
// CHECK: hmx.pack_act
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
func.func @mask_dot_chain(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                          %v: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>

  // One materialised arange: the only `linalg.index` in the file (as `tl.arange`
  // leaves it), so this map is a hard stop for the propagation.
  %e64 = tensor.empty() : tensor<64xi32>
  %iv = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>],
                        iterator_types = ["parallel"]}
      outs(%e64 : tensor<64xi32>) {
  ^bb0(%o: i32):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i32
    linalg.yield %ii : i32
  } -> tensor<64xi32>

  // broadcast to rows: (i, j) -> (i)
  %er = tensor.empty() : tensor<64x64xi32>
  %row = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%iv : tensor<64xi32>) outs(%er : tensor<64x64xi32>) {
  ^bb0(%x: i32, %o: i32):
    linalg.yield %x : i32
  } -> tensor<64x64xi32>

  // broadcast to columns: (i, j) -> (j)
  %ec = tensor.empty() : tensor<64x64xi32>
  %col = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%iv : tensor<64xi32>) outs(%ec : tensor<64x64xi32>) {
  ^bb0(%x: i32, %o: i32):
    linalg.yield %x : i32
  } -> tensor<64x64xi32>

  %ei = tensor.empty() : tensor<64x64xi1>
  %cmp = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%row, %col : tensor<64x64xi32>, tensor<64x64xi32>)
      outs(%ei : tensor<64x64xi1>) {
  ^bb0(%r: i32, %c2: i32, %o: i1):
    %p = arith.cmpi sge, %r, %c2 : i32
    linalg.yield %p : i1
  } -> tensor<64x64xi1>

  // The f16 select: it stays row-major with the rest of the mask chain
  // (`select` is not cheap f16 arithmetic), reading the unpacked read-out.
  %em = tensor.empty() : tensor<64x64xf16>
  %masked = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%cmp, %m0 : tensor<64x64xi1>, tensor<64x64xf16>) outs(%em : tensor<64x64xf16>) {
  ^bb0(%m: i1, %x: f16, %o: f16):
    %z = arith.constant 0.000000e+00 : f16
    %s = arith.select %m, %x, %z : f16
    linalg.yield %s : f16
  } -> tensor<64x64xf16>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %z2 = arith.constant 0.000000e+00 : f16
  %i2 = linalg.fill ins(%z2 : f16) outs(%e2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%masked, %v : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%i2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}

//----------------------------------------------------------------------------//
// Negative: a reduction is the co-scheduling boundary.
//
// The row-sum spans croutons and has no crouton representation, so one
// conversion must remain in front of it.  A propagation that folded the masked
// scores all the way into the reduce would be over-folding: `hmx.matmul` feeds
// a crouton, and the reduce still needs its row-major image.
//----------------------------------------------------------------------------//
// CHECK-LABEL: func.func @mask_then_reduce
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
// CHECK: linalg.reduce
// CHECK-NOT: hmx.matmul
// CHECK-NOT: hmx.pack_act
func.func @mask_then_reduce(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                            %mask: tensor<64x64xi1>) -> tensor<64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %masked = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mask, %m0 : tensor<64x64xi1>, tensor<64x64xf16>) outs(%e1 : tensor<64x64xf16>) {
  ^bb0(%m: i1, %x: f16, %o: f16):
    %z = arith.constant 0.000000e+00 : f16
    %s = arith.select %m, %x, %z : f16
    linalg.yield %s : f16
  } -> tensor<64x64xf16>
  %er = tensor.empty() : tensor<64xf16>
  %red = linalg.reduce ins(%masked : tensor<64x64xf16>) outs(%er : tensor<64xf16>)
      dimensions = [1] (%x: f16, %acc: f16) {
    %s = arith.addf %x, %acc : f16
    linalg.yield %s : f16
  }
  return %red : tensor<64xf16>
}

//----------------------------------------------------------------------------//
// Positive: a chain of cheap f16 maps still folds.
//
// `mulf` then `addf` (f16 only, splat constants in the body) commute with the
// crouton permutation, so the read-out of the first matmul stays in the
// crouton layout, both maps run on it as one fused generic, and the second
// matmul consumes the result directly -- no `hmx.unpack_acc` and no
// `hmx.pack_act` between the two.
//----------------------------------------------------------------------------//
// CHECK-LABEL: func.func @cheap_scale_chain
// CHECK: hmx.matmul
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: hmx.pack_act
// CHECK: linalg.generic
// CHECK-SAME: tensor<2x2x16x32x2xf16>
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: hmx.pack_act
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
func.func @cheap_scale_chain(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                             %c: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %s1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%m0 : tensor<64x64xf16>) outs(%e1 : tensor<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %half = arith.constant 5.000000e-01 : f16
    %mul = arith.mulf %in, %half : f16
    linalg.yield %mul : f16
  } -> tensor<64x64xf16>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %s2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%s1 : tensor<64x64xf16>) outs(%e2 : tensor<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %quarter = arith.constant 2.500000e-01 : f16
    %add = arith.addf %in, %quarter : f16
    linalg.yield %add : f16
  } -> tensor<64x64xf16>
  %e3 = tensor.empty() : tensor<64x64xf16>
  %z3 = arith.constant 0.000000e+00 : f16
  %i3 = linalg.fill ins(%z3 : f16) outs(%e3 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%s2, %c : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i3 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}

//----------------------------------------------------------------------------//
// FA three-stage boundary: dot1's read-out feeds an f32 rowsum, and the
// normalised scores feed dot2. The HVX middle (widen/rowsum/normalise/narrow)
// is row-major f32 work, and a reduction spans croutons, so the whole-tree
// veto keeps BOTH conversions: dot1's unpack (its epilogue) and dot2's pack
// (its prologue). Folding either side would be over-folding: this is the
// co-scheduling point, not a compromise.
//----------------------------------------------------------------------------//
// CHECK-LABEL: func.func @rowsum_boundary
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
// CHECK: arith.extf
// CHECK: linalg.reduce
// CHECK: arith.divf
// CHECK: arith.truncf
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
func.func @rowsum_boundary(%q: tensor<64x64xf16>, %k: tensor<64x64xf16>,
                           %v: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %s = linalg.matmul ins(%q, %k : tensor<64x64xf16>, tensor<64x64xf16>)
                    outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %ef = tensor.empty() : tensor<64x64xf32>
  %sf = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%s : tensor<64x64xf16>) outs(%ef : tensor<64x64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %x = arith.extf %in : f16 to f32
    linalg.yield %x : f32
  } -> tensor<64x64xf32>
  %er = tensor.empty() : tensor<64xf32>
  %zf = arith.constant 0.000000e+00 : f32
  %ir = linalg.fill ins(%zf : f32) outs(%er : tensor<64xf32>) -> tensor<64xf32>
  %rowsum = linalg.reduce ins(%sf : tensor<64x64xf32>) outs(%ir : tensor<64xf32>)
      dimensions = [1] (%x: f32, %acc: f32) {
    %a = arith.addf %x, %acc : f32
    linalg.yield %a : f32
  }
  %en = tensor.empty() : tensor<64x64xf32>
  %p = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%sf, %rowsum : tensor<64x64xf32>, tensor<64xf32>) outs(%en : tensor<64x64xf32>) {
  ^bb0(%x: f32, %r: f32, %o: f32):
    %d = arith.divf %x, %r : f32
    linalg.yield %d : f32
  } -> tensor<64x64xf32>
  %eh = tensor.empty() : tensor<64x64xf16>
  %ph = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%p : tensor<64x64xf32>) outs(%eh : tensor<64x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %z1 = arith.constant 0.000000e+00 : f16
  %i1 = linalg.fill ins(%z1 : f16) outs(%e1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %o = linalg.matmul ins(%ph, %v : tensor<64x64xf16>, tensor<64x64xf16>)
                    outs(%i1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %o : tensor<64x64xf16>
}
