//===- matmul-to-hexkl-u8i8.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(matmul-to-hexkl{mode=micro}))' \
// RUN:   | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(matmul-to-hexkl{mode=macro}))' \
// RUN:   | FileCheck %s --check-prefix=MACRO
// RUN: linalg-hexagon-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(matmul-to-hexkl{mode=micro}))' \
// RUN:   | FileCheck %s --check-prefix=REUSE

// HMX mm_u8i8 uses unsigned activations and signed weights. Signed i8 dot lowers
// exactly as mm_u8i8(A ^ 0x80, B) - 128 * colsum(B).

// -----

// i8 x i8 -> i32, micro mode: bias, colsum, hexkl.matmul, correction.
func.func @matmul_i8(%a: tensor<64x32xi8>, %b: tensor<32x32xi8>) -> tensor<64x32xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<64x32xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<64x32xi32>) -> tensor<64x32xi32>
  %r = linalg.matmul ins(%a, %b : tensor<64x32xi8>, tensor<32x32xi8>)
                     outs(%z : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %r : tensor<64x32xi32>
}

// CHECK-LABEL: func.func @matmul_i8
// Use xori with -128 (0x80), allowing an existing producer-side shift to cancel.
// CHECK-DAG:   %[[SIGNBIT:.*]] = arith.constant -128 : i8
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : i32
// CHECK:       %[[BIASED:.*]] = linalg.generic
// CHECK-SAME:    ins(%arg0 : tensor<64x32xi8>)
// CHECK:         arith.xori %{{.*}}, %[[SIGNBIT]] : i8
// colsum reduces K (the weight's dim 0) into a length-N i32 vector, widening
// each i8 before accumulating so a large K cannot wrap.
// CHECK:       %[[CSINIT:.*]] = linalg.fill
// CHECK:       %[[COLSUM:.*]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%arg1 : tensor<32x32xi8>)
// CHECK:         arith.extsi %{{.*}} : i8 to i32
// CHECK:         arith.addi
// The matmul takes the BIASED activation, not the original, and the original
// (signed) weight.
// CHECK:       %[[RAW:.*]] = hexkl.matmul ins(%[[BIASED]], %arg1 : tensor<64x32xi8>, tensor<32x32xi8>)
// Correction subtracts 128*colsum[n], broadcast across m (colsum's map drops d0).
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[RAW]], %[[COLSUM]] : tensor<64x32xi32>, tensor<32xi32>)
// CHECK:         arith.muli %{{.*}}, %[[C128]] : i32
// CHECK:         arith.subi

// Macro mode has no u8i8 entry point (hexkl_macro_* is f16-only), so the i8
// matmul must be left alone rather than converted.
// MACRO-LABEL: func.func @matmul_i8
// MACRO:       linalg.matmul
// MACRO-NOT:   hexkl.matmul

// -----

// f16 is unchanged by the i8 support: still a bare hexkl.matmul with no bias
// and no correction.
func.func @matmul_f16(%a: tensor<32x32xf16>, %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
  %e = tensor.empty() : tensor<32x32xf32>
  %r = linalg.matmul ins(%a, %b : tensor<32x32xf16>, tensor<32x32xf16>)
                     outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @matmul_f16
// CHECK:       hexkl.matmul ins(%arg0, %arg1 : tensor<32x32xf16>, tensor<32x32xf16>)
// CHECK-NOT:   arith.xori
// CHECK-NOT:   arith.extsi

// -----

// Mixed element types are rejected: HMX needs both operands the same width, and
// hexkl.matmul's verifier only admits (f16,f16) or (i8,i8).
func.func @matmul_mixed(%a: tensor<64x32xf16>, %b: tensor<32x32xi8>,
                        %z: tensor<64x32xi32>) -> tensor<64x32xi32> {
  %r = linalg.matmul ins(%a, %b : tensor<64x32xf16>, tensor<32x32xi8>)
                     outs(%z : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %r : tensor<64x32xi32>
}

// CHECK-LABEL: func.func @matmul_mixed
// CHECK:       linalg.matmul
// CHECK-NOT:   hexkl.matmul

// -----

// The u8i8 weight-layout API reads complete K=32 blocks; a K tail would read
// past the row-major weight tensor, so this remains a linalg.matmul.
func.func @matmul_i8_k_tail(%a: tensor<64x40xi8>, %b: tensor<40x32xi8>,
                            %z: tensor<64x32xi32>) -> tensor<64x32xi32> {
  %r = linalg.matmul ins(%a, %b : tensor<64x40xi8>, tensor<40x32xi8>)
                     outs(%z : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %r : tensor<64x32xi32>
}

// CHECK-LABEL: func.func @matmul_i8_k_tail
// CHECK:       linalg.matmul
// CHECK-NOT:   hexkl.matmul

// -----

// Reject the N tail; keep K aligned to isolate this check.
func.func @matmul_i8_n_tail(%a: tensor<64x32xi8>, %b: tensor<32x40xi8>,
                            %z: tensor<64x40xi32>) -> tensor<64x40xi32> {
  %r = linalg.matmul ins(%a, %b : tensor<64x32xi8>, tensor<32x40xi8>)
                     outs(%z : tensor<64x40xi32>) -> tensor<64x40xi32>
  return %r : tensor<64x40xi32>
}

// CHECK-LABEL: func.func @matmul_i8_n_tail
// CHECK:       linalg.matmul
// CHECK-NOT:   hexkl.matmul

// -----

// i8 x i8 with a too-narrow accumulator is rejected: the HMX integer
// accumulator is i32, and acc_read_int32 writes i32 tiles.
func.func @matmul_i8_i16_acc(%a: tensor<64x32xi8>, %b: tensor<32x32xi8>,
                             %z: tensor<64x32xi16>) -> tensor<64x32xi16> {
  %r = linalg.matmul ins(%a, %b : tensor<64x32xi8>, tensor<32x32xi8>)
                     outs(%z : tensor<64x32xi16>) -> tensor<64x32xi16>
  return %r : tensor<64x32xi16>
}

// CHECK-LABEL: func.func @matmul_i8_i16_acc
// CHECK:       linalg.matmul
// CHECK-NOT:   hexkl.matmul

// -----

// If the RHS already has an exact centered-i8 producer and its i32 form is
// reduced over K, the pass must reuse that reduction instead of emitting a
// second reduction over the truncated i8 tensor.
func.func @matmul_i8_reuse_existing_colsum(
    %a: tensor<64x32xi8>, %b_u8: tensor<32x32xi8>)
    -> (tensor<64x32xi32>, tensor<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %wide_empty = tensor.empty() : tensor<32x32xi32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%b_u8 : tensor<32x32xi8>) outs(%wide_empty : tensor<32x32xi32>) {
    ^bb0(%in: i8, %out: i32):
      %u = arith.extui %in : i8 to i32
      linalg.yield %u : i32
  } -> tensor<32x32xi32>
  %center_empty = tensor.empty() : tensor<32x32xi32>
  %center_init = linalg.fill ins(%c128 : i32) outs(%center_empty : tensor<32x32xi32>)
      -> tensor<32x32xi32>
  %center = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%wide, %center_init : tensor<32x32xi32>, tensor<32x32xi32>)
      outs(%center_empty : tensor<32x32xi32>) {
    ^bb0(%in: i32, %unused: i32, %out: i32):
      %centered = arith.subi %in, %unused : i32
      linalg.yield %centered : i32
  } -> tensor<32x32xi32>
  %sum_empty = tensor.empty() : tensor<32xi32>
  %sum_init = linalg.fill ins(%c0 : i32) outs(%sum_empty : tensor<32xi32>)
      -> tensor<32xi32>
  %sum = linalg.reduce ins(%center : tensor<32x32xi32>)
      outs(%sum_init : tensor<32xi32>) dimensions = [0]
      (%in: i32, %init: i32) {
    %added = arith.addi %in, %init : i32
    linalg.yield %added : i32
  }
  %narrow_empty = tensor.empty() : tensor<32x32xi8>
  %narrow = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%center : tensor<32x32xi32>) outs(%narrow_empty : tensor<32x32xi8>) {
    ^bb0(%in: i32, %out: i8):
      %narrowed = arith.trunci %in : i32 to i8
      linalg.yield %narrowed : i8
  } -> tensor<32x32xi8>
  %out_empty = tensor.empty() : tensor<64x32xi32>
  %out = linalg.matmul ins(%a, %narrow : tensor<64x32xi8>, tensor<32x32xi8>)
      outs(%out_empty : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %out, %sum : tensor<64x32xi32>, tensor<32xi32>
}

// CHECK-LABEL: func.func @matmul_i8_reuse_existing_colsum
// REUSE-LABEL: func.func @matmul_i8_reuse_existing_colsum
// REUSE:       linalg.reduce
// REUSE-NOT:   iterator_types = ["parallel", "reduction"]
// REUSE:       hexkl.matmul

// -----

// A nonzero reduction initializer is not a column sum and must not be reused.
// CHECK-LABEL: func.func @matmul_i8_nonzero_colsum_init
// CHECK:       linalg.generic
// CHECK:       iterator_types = ["parallel", "reduction"]
func.func @matmul_i8_nonzero_colsum_init(
    %a: tensor<64x32xi8>, %b_u8: tensor<32x32xi8>) -> tensor<64x32xi32> {
  %c1 = arith.constant 1 : i32
  %wide_empty = tensor.empty() : tensor<32x32xi32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%b_u8 : tensor<32x32xi8>) outs(%wide_empty : tensor<32x32xi32>) {
    ^bb0(%in: i8, %out: i32):
      %u = arith.extui %in : i8 to i32
      linalg.yield %u : i32
  } -> tensor<32x32xi32>
  %sum_empty = tensor.empty() : tensor<32xi32>
  %sum_init = linalg.fill ins(%c1 : i32) outs(%sum_empty : tensor<32xi32>) -> tensor<32xi32>
  %sum = linalg.reduce ins(%wide : tensor<32x32xi32>) outs(%sum_init : tensor<32xi32>) dimensions = [0]
      (%in: i32, %init: i32) {
    %added = arith.addi %in, %init : i32
    linalg.yield %added : i32
  }
  %narrow_empty = tensor.empty() : tensor<32x32xi8>
  %narrow = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<32x32xi32>) outs(%narrow_empty : tensor<32x32xi8>) {
    ^bb0(%in: i32, %out: i8):
      %v = arith.trunci %in : i32 to i8
      linalg.yield %v : i8
  } -> tensor<32x32xi8>
  %out_empty = tensor.empty() : tensor<64x32xi32>
  %out = linalg.matmul ins(%a, %narrow : tensor<64x32xi8>, tensor<32x32xi8>) outs(%out_empty : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %out : tensor<64x32xi32>
}

// -----

// A reduction after the matmul cannot supply its correction value.
// CHECK-LABEL: func.func @matmul_i8_late_colsum
// CHECK:       iterator_types = ["parallel", "reduction"]
// CHECK:       hexkl.matmul
func.func @matmul_i8_late_colsum(
    %a: tensor<64x32xi8>, %b_u8: tensor<32x32xi8>) -> tensor<64x32xi32> {
  %wide_empty = tensor.empty() : tensor<32x32xi32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%b_u8 : tensor<32x32xi8>) outs(%wide_empty : tensor<32x32xi32>) {
    ^bb0(%in: i8, %out: i32):
      %u = arith.extui %in : i8 to i32
      linalg.yield %u : i32
  } -> tensor<32x32xi32>
  %narrow_empty = tensor.empty() : tensor<32x32xi8>
  %narrow = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<32x32xi32>) outs(%narrow_empty : tensor<32x32xi8>) {
    ^bb0(%in: i32, %out: i8):
      %v = arith.trunci %in : i32 to i8
      linalg.yield %v : i8
  } -> tensor<32x32xi8>
  %out_empty = tensor.empty() : tensor<64x32xi32>
  %out = linalg.matmul ins(%a, %narrow : tensor<64x32xi8>, tensor<32x32xi8>) outs(%out_empty : tensor<64x32xi32>) -> tensor<64x32xi32>
  %c0 = arith.constant 0 : i32
  %sum_empty = tensor.empty() : tensor<32xi32>
  %sum_init = linalg.fill ins(%c0 : i32) outs(%sum_empty : tensor<32xi32>) -> tensor<32xi32>
  %sum = linalg.reduce ins(%wide : tensor<32x32xi32>) outs(%sum_init : tensor<32xi32>) dimensions = [0]
      (%in: i32, %init: i32) {
    %added = arith.addi %in, %init : i32
    linalg.yield %added : i32
  }
  return %out : tensor<64x32xi32>
}

// -----

// Only an i8-to-i32 unsigned extension matches the centered-i8 producer.
// CHECK-LABEL: func.func @matmul_i8_extui_i16
// CHECK:       linalg.generic
// CHECK:       iterator_types = ["parallel", "reduction"]
func.func @matmul_i8_extui_i16(
    %a: tensor<64x32xi8>, %b_i16: tensor<32x32xi16>) -> tensor<64x32xi32> {
  %wide_empty = tensor.empty() : tensor<32x32xi32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%b_i16 : tensor<32x32xi16>) outs(%wide_empty : tensor<32x32xi32>) {
    ^bb0(%in: i16, %out: i32):
      %u = arith.extui %in : i16 to i32
      linalg.yield %u : i32
  } -> tensor<32x32xi32>
  %narrow_empty = tensor.empty() : tensor<32x32xi8>
  %narrow = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<32x32xi32>) outs(%narrow_empty : tensor<32x32xi8>) {
    ^bb0(%in: i32, %out: i8):
      %v = arith.trunci %in : i32 to i8
      linalg.yield %v : i8
  } -> tensor<32x32xi8>
  %c0 = arith.constant 0 : i32
  %sum_empty = tensor.empty() : tensor<32xi32>
  %sum_init = linalg.fill ins(%c0 : i32) outs(%sum_empty : tensor<32xi32>) -> tensor<32xi32>
  %sum = linalg.reduce ins(%wide : tensor<32x32xi32>) outs(%sum_init : tensor<32xi32>) dimensions = [0]
      (%in: i32, %init: i32) {
    %added = arith.addi %in, %init : i32
    linalg.yield %added : i32
  }
  %out_empty = tensor.empty() : tensor<64x32xi32>
  %out = linalg.matmul ins(%a, %narrow : tensor<64x32xi8>, tensor<32x32xi8>) outs(%out_empty : tensor<64x32xi32>) -> tensor<64x32xi32>
  return %out : tensor<64x32xi32>
}
