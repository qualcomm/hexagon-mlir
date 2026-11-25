//===- hexagon_vectorize_packed_generic.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt --hexagon-vectorize-packed-generic %s | FileCheck %s --check-prefixes=CHECK,COMMON,SPLIT,NOEAGER
// RUN: linalg-hexagon-opt --hexagon-vectorize-packed-generic="enable-packed-generic-loop-splitting=false" %s | FileCheck %s --check-prefixes=NOSPLIT,COMMON,NOEAGER
// RUN: linalg-hexagon-opt --hexagon-vectorize-packed-generic="enable-packed-generic-eager-vectorization=true" %s | FileCheck %s --check-prefixes=EAGER,COMMON,SPLIT

// Consolidated tests for HexagonVectorizePackedGenericPass.
//
// Sections:
//   1. Reduction bodies (add, mul, max, sum-of-squares, scalar input)
//   2. Parallel - subtract mean (map10 broadcast input, divf+subf body)
//   3. Parallel - bias add (map6 bias, addf body)
//   4. Parallel - simple GeLU-like (map6 bias + scalar tensor inputs, tanh body)
//   5. Parallel - tricky normalization (map6 bias with (d1,d4) indexing, rsqrt body)
//   6. Dynamic shapes (regression - bail out, do not crash)

// Shared affine maps
#map5  = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map6  = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>

// ============================================================================
// Section 1: Reduction bodies
//
// These tests verify that:
//   1. The body data flow is followed.
//   2. Different reduction bodies (add, mul, max) are handled correctly.
//   3. Multi-op bodies (e.g., sum-of-squares) are handled correctly.
//   4. Scalar inputs stay scalar until a vector operand forces broadcast.
//   5. The vector accumulator is initialized with the correct identity element.
// ============================================================================

// ----------------------------------------------------------------------------
// Test 1: Reduction with addf body - verifies body data flow is followed.
// ----------------------------------------------------------------------------

// COMMON-LABEL: func.func @test_reduction_add
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16>
// COMMON-DAG:   %[[VINIT:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:             %[[VACC:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON-SAME:            iter_args(%[[ITER:.*]] = %[[VINIT]]) -> (vector<64xf16>)
// COMMON:               %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// COMMON:               %[[VSUM:.*]] = arith.addf %[[VIN]], %[[ITER]] : vector<64xf16>
// COMMON-NOT:            arith.mulf
// COMMON:               scf.yield %[[VSUM]] : vector<64xf16>
// COMMON:             %[[EVEN:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [0, 2, 4, 6
// COMMON:             %[[R0:.*]] = vector.reduction <add>, %[[EVEN]] : vector<32xf16> into f16
// COMMON:             %[[INS0:.*]] = tensor.insert %[[R0]] into %{{.*}}[{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// COMMON:             %[[ODD:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [1, 3, 5, 7
// COMMON:             %[[R1:.*]] = vector.reduction <add>, %[[ODD]] : vector<32xf16> into f16
// COMMON:             tensor.insert %[[R1]] into %[[INS0]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>

func.func @test_reduction_add(%arg0: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16> {
  %c0 = arith.constant 0.0 : f16
  %init0 = tensor.empty() : tensor<2x8x2x2xf16>
  %init = linalg.fill ins(%c0 : f16) outs(%init0 : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map10],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
  } ins(%arg0 : tensor<2x24x8x2x32x2xf16>) outs(%init : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
  } -> tensor<2x8x2x2xf16>
  return %res : tensor<2x8x2x2xf16>
}

// ----------------------------------------------------------------------------
// Test 2: Reduction with mulf body - verifies MUL combining kind.
// The vector accumulator must be initialized to 1.0 (identity for mul),
// and vector.reduction <mul> must be used.
// ----------------------------------------------------------------------------

// COMMON-LABEL: func.func @test_reduction_mul
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16>
// COMMON-DAG:   %[[VINIT:.*]] = arith.constant dense<1.000000e+00> : vector<64xf16>
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:             %[[VACC:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON-SAME:            iter_args(%[[ITER:.*]] = %[[VINIT]]) -> (vector<64xf16>)
// COMMON:               %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// COMMON:               %[[VPROD:.*]] = arith.mulf %[[VIN]], %[[ITER]] : vector<64xf16>
// COMMON:               scf.yield %[[VPROD]] : vector<64xf16>
// COMMON:             %[[EVEN:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [0, 2, 4, 6
// COMMON:             %[[R0:.*]] = vector.reduction <mul>, %[[EVEN]] : vector<32xf16> into f16
// COMMON:             %[[INS0:.*]] = tensor.insert %[[R0]] into %{{.*}}[{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// COMMON:             %[[ODD:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [1, 3, 5, 7
// COMMON:             %[[R1:.*]] = vector.reduction <mul>, %[[ODD]] : vector<32xf16> into f16
// COMMON:             tensor.insert %[[R1]] into %[[INS0]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>

func.func @test_reduction_mul(%arg0: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16> {
  %c1 = arith.constant 1.0 : f16
  %init0 = tensor.empty() : tensor<2x8x2x2xf16>
  %init = linalg.fill ins(%c1 : f16) outs(%init0 : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map10],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
  } ins(%arg0 : tensor<2x24x8x2x32x2xf16>) outs(%init : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %prod = arith.mulf %in, %out : f16
      linalg.yield %prod : f16
  } -> tensor<2x8x2x2xf16>
  return %res : tensor<2x8x2x2xf16>
}

// ----------------------------------------------------------------------------
// Test 3: Reduction with maximumf body - verifies MAXIMUMF combining kind.
// The vector accumulator must be initialized to -inf (identity for max),
// and vector.reduction <maximumf> must be used.
// ----------------------------------------------------------------------------

// COMMON-LABEL: func.func @test_reduction_max
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16>
// COMMON-DAG:   %[[VINIT:.*]] = arith.constant dense<0xFC00> : vector<64xf16>
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:             %[[VACC:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON-SAME:            iter_args(%[[ITER:.*]] = %[[VINIT]]) -> (vector<64xf16>)
// COMMON:               %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// COMMON:               %[[VMAX:.*]] = arith.maximumf %[[VIN]], %[[ITER]] : vector<64xf16>
// COMMON:               scf.yield %[[VMAX]] : vector<64xf16>
// COMMON:             %[[EVEN:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [0, 2, 4, 6
// COMMON:             %[[R0:.*]] = vector.reduction <maximumf>, %[[EVEN]] : vector<32xf16> into f16
// COMMON:             %[[INS0:.*]] = tensor.insert %[[R0]] into %{{.*}}[{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// COMMON:             %[[ODD:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [1, 3, 5, 7
// COMMON:             %[[R1:.*]] = vector.reduction <maximumf>, %[[ODD]] : vector<32xf16> into f16
// COMMON:             tensor.insert %[[R1]] into %[[INS0]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>

func.func @test_reduction_max(%arg0: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16> {
  %cst = arith.constant 0xFF80 : f16
  %init0 = tensor.empty() : tensor<2x8x2x2xf16>
  %init = linalg.fill ins(%cst : f16) outs(%init0 : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map10],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
  } ins(%arg0 : tensor<2x24x8x2x32x2xf16>) outs(%init : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %max = arith.maximumf %in, %out : f16
      linalg.yield %max : f16
  } -> tensor<2x8x2x2xf16>
  return %res : tensor<2x8x2x2xf16>
}

// ----------------------------------------------------------------------------
// Test 4: Reduction with multi-op body (sum of squares: mulf then addf).
// Verifies that multi-op bodies are correctly cloned following data flow.
// ----------------------------------------------------------------------------

// COMMON-LABEL: func.func @test_reduction_sum_of_squares
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16>
// COMMON-DAG:   %[[VINIT:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:             %[[VACC:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON-SAME:            iter_args(%[[ITER:.*]] = %[[VINIT]]) -> (vector<64xf16>)
// COMMON:               %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// Squaring: both operands of mulf must be the same loaded vector, not the accumulator.
// COMMON:               %[[VSQ:.*]] = arith.mulf %[[VIN]], %[[VIN]] : vector<64xf16>
// COMMON:               %[[VADD:.*]] = arith.addf %[[VSQ]], %[[ITER]] : vector<64xf16>
// COMMON:               scf.yield %[[VADD]] : vector<64xf16>
// COMMON:             %[[EVEN:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [0, 2, 4, 6
// COMMON:             %[[R0:.*]] = vector.reduction <add>, %[[EVEN]] : vector<32xf16> into f16
// COMMON:             %[[INS0:.*]] = tensor.insert %[[R0]] into %{{.*}}[{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// COMMON:             %[[ODD:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [1, 3, 5, 7
// COMMON:             %[[R1:.*]] = vector.reduction <add>, %[[ODD]] : vector<32xf16> into f16
// COMMON:             tensor.insert %[[R1]] into %[[INS0]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>

func.func @test_reduction_sum_of_squares(%arg0: tensor<2x24x8x2x32x2xf16>) -> tensor<2x8x2x2xf16> {
  %c0 = arith.constant 0.0 : f16
  %init0 = tensor.empty() : tensor<2x8x2x2xf16>
  %init = linalg.fill ins(%c0 : f16) outs(%init0 : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map10],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
  } ins(%arg0 : tensor<2x24x8x2x32x2xf16>) outs(%init : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sq = arith.mulf %in, %in : f16
      %sum = arith.addf %sq, %out : f16
      linalg.yield %sum : f16
  } -> tensor<2x8x2x2xf16>
  return %res : tensor<2x8x2x2xf16>
}

// ----------------------------------------------------------------------------
// Test 5: Reduction with non-packed scalar input.
// Verifies that the scalar input (%scale) is extracted as a scalar
// (tensor.extract), and only broadcast to vector when the body op (mulf)
// requires it (because the other operand is a vector).
// ----------------------------------------------------------------------------

// COMMON-LABEL: func.func @test_reduction_with_scalar_input
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>, %[[ARG1:.*]]: tensor<2x8xf16>) -> tensor<2x8x2x2xf16>
// COMMON-DAG:   %[[VINIT:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:             %[[VACC:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON-SAME:            iter_args(%[[ITER:.*]] = %[[VINIT]]) -> (vector<64xf16>)
// COMMON:               %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// Scalar input: extracted as f16, then broadcast to vector only when needed for mulf.
// COMMON:               %[[SCALE:.*]] = tensor.extract %[[ARG1]][{{.*}}] : tensor<2x8xf16>
// COMMON:               %[[VSCALE:.*]] = vector.broadcast %[[SCALE]] : f16 to vector<64xf16>
// COMMON:               %[[VPROD:.*]] = arith.mulf %[[VIN]], %[[VSCALE]] : vector<64xf16>
// COMMON:               %[[VSUM:.*]] = arith.addf %[[VPROD]], %[[ITER]] : vector<64xf16>
// COMMON:               scf.yield %[[VSUM]] : vector<64xf16>
// COMMON:             %[[EVEN:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [0, 2, 4, 6
// COMMON:             %[[R0:.*]] = vector.reduction <add>, %[[EVEN]] : vector<32xf16> into f16
// COMMON:             %[[INS0:.*]] = tensor.insert %[[R0]] into %{{.*}}[{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// COMMON:             %[[ODD:.*]] = vector.shuffle %[[VACC]], %[[VACC]] [1, 3, 5, 7
// COMMON:             %[[R1:.*]] = vector.reduction <add>, %[[ODD]] : vector<32xf16> into f16
// COMMON:             tensor.insert %[[R1]] into %[[INS0]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>

#map_in1_scalar = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2)>

func.func @test_reduction_with_scalar_input(%arg0: tensor<2x24x8x2x32x2xf16>, %arg1: tensor<2x8xf16>) -> tensor<2x8x2x2xf16> {
  %c0 = arith.constant 0.0 : f16
  %init0 = tensor.empty() : tensor<2x8x2x2xf16>
  %init = linalg.fill ins(%c0 : f16) outs(%init0 : tensor<2x8x2x2xf16>) -> tensor<2x8x2x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map_in1_scalar, #map10],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x24x8x2x32x2xf16>, tensor<2x8xf16>) outs(%init : tensor<2x8x2x2xf16>) {
    ^bb0(%in: f16, %scale: f16, %out: f16):
      %scaled = arith.mulf %in, %scale : f16
      %sum = arith.addf %scaled, %out : f16
      linalg.yield %sum : f16
  } -> tensor<2x8x2x2xf16>
  return %res : tensor<2x8x2x2xf16>
}

// ============================================================================
// Section 2: Parallel - subtract mean (map10 broadcast input, divf+subf body)
//
// Tests Op C from gpt2_slice.mlir: subtract mean (normalize step).
// Maps: [map5, map10] -> map5 (all-parallel 6D, map10 input broadcast)
// Body: out = in - (sum / 768.0)
// Unique: all-parallel with map10 broadcast input, divf+subf body,
//         broadcasts to vector<32xf16>.
// ============================================================================

// COMMON-LABEL: func.func @subtract_mean
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>, %[[ARG1:.*]]: tensor<2x8x2x2xf16>)
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// COMMON-DAG:   %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED_IN:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:         scf.for %{{.*}} = %[[C0]] to %[[C24]] step %[[C1]]
// COMMON:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// COMMON:             scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// COMMON:               %[[E0:.*]] = tensor.extract %[[ARG1]]{{.*}} : tensor<2x8x2x2xf16>
// COMMON:               %[[E1:.*]] = tensor.extract %[[ARG1]]{{.*}} : tensor<2x8x2x2xf16>
// Mode-specific: NOEAGER does scalar divf then broadcast; EAGER broadcasts first then vector divf.
// Data flow for default/nosplit: extract → divf → transfer_read → broadcast → shuffle → subf
// NOEAGER:              %[[D0:.*]] = arith.divf %[[E0]], %{{.*}} : f16
// NOEAGER:              %[[D1:.*]] = arith.divf %[[E1]], %{{.*}} : f16
// NOEAGER:              %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED_IN]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// NOEAGER:              %[[B0:.*]] = vector.broadcast %[[D0]] : f16 to vector<32xf16>
// NOEAGER:              %[[B1:.*]] = vector.broadcast %[[D1]] : f16 to vector<32xf16>
// NOEAGER:              %[[INTERLEAVED:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 32
// NOEAGER:              %[[RESULT:.*]] = arith.subf %[[VIN]], %[[INTERLEAVED]] : vector<64xf16>
// NOEAGER:              vector.transfer_write %[[RESULT]], %{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
// Data flow for eager: extract → broadcast → shuffle → divf → transfer_read → subf
// EAGER:                %[[B0:.*]] = vector.broadcast %[[E0]] : f16 to vector<32xf16>
// EAGER:                %[[B1:.*]] = vector.broadcast %[[E1]] : f16 to vector<32xf16>
// EAGER:                %[[SHUFFLED:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 32
// EAGER:                %[[INTERLEAVED:.*]] = arith.divf %[[SHUFFLED]], %{{.*}} : vector<64xf16>
// EAGER:                %[[VIN:.*]] = vector.transfer_read %[[COLLAPSED_IN]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// EAGER:                %[[RESULT:.*]] = arith.subf %[[VIN]], %[[INTERLEAVED]] : vector<64xf16>
// EAGER:                vector.transfer_write %[[RESULT]], %{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
// COMMON:       tensor.expand_shape %{{.*}} {{.*}} : tensor<2x24x8x2x64xf16> into tensor<2x24x8x2x32x2xf16>

func.func @subtract_mean(%arg0: tensor<2x24x8x2x32x2xf16>,
                          %arg1: tensor<2x8x2x2xf16>) -> tensor<2x24x8x2x32x2xf16> {
  %init = tensor.empty() : tensor<2x24x8x2x32x2xf16>
  %cst = arith.constant 7.680000e+02 : f16
  %res = linalg.generic {
    indexing_maps = [#map5, #map10, #map5],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x24x8x2x32x2xf16>, tensor<2x8x2x2xf16>)
    outs(%init : tensor<2x24x8x2x32x2xf16>) {
  ^bb0(%in: f16, %in_136: f16, %out: f16):
    %0 = arith.divf %in_136, %cst : f16
    %1 = arith.subf %in, %0 : f16
    linalg.yield %1 : f16
  } -> tensor<2x24x8x2x32x2xf16>
  return %res : tensor<2x24x8x2x32x2xf16>
}

// ============================================================================
// Section 3: Parallel - bias add (map6 bias, addf body)
//
// Tests Op F-simple from gpt2_slice.mlir: simple bias add.
// Maps: [map5, map6] -> map5 (all-parallel 6D, map6 bias broadcast)
// Body: out = in + bias
// Unique: map6 bias requiring expand_shape+collapse reshape; plain addf body
//         (the GeLU variant of the same maps is in Section 4).
// ============================================================================

// COMMON-LABEL: func.func @bias_add
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>, %[[ARG1:.*]]: tensor<24x32xf16>)
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// SPLIT-DAG:    %[[C12:.*]] = arith.constant 12 : index
// NOSPLIT-DAG:  %[[C24:.*]] = arith.constant 24 : index
// COMMON:       %[[COLLAPSED_IN:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       %[[BIAS_EXP:.*]] = tensor.expand_shape %[[ARG1]] {{.*}} : tensor<24x32xf16> into tensor<12x2x32xf16>
// COMMON:       %[[BIAS_COL:.*]] = tensor.collapse_shape %[[BIAS_EXP]] {{.*}} : tensor<12x2x32xf16> into tensor<12x64xf16>
//
// Loop-split path (SPLIT) — inner loop runs over 12 entries with muli/addi.
// SPLIT:        scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// SPLIT:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// SPLIT:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// SPLIT:              scf.for %[[IV:.*]] = %[[C0]] to %[[C12]] step %[[C1]] iter_args
// SPLIT:                %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// SPLIT:                %[[HALF0:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [0, 0, 1, 1
// SPLIT:                %[[HALF1:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [32, 32, 33, 33
// SPLIT:                %[[IDX0:.*]] = arith.muli %[[IV]], %[[C2]] : index
// SPLIT:                %[[IN0:.*]] = vector.transfer_read %[[COLLAPSED_IN]][%{{.*}}, %[[IDX0]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// SPLIT:                %[[OUT0:.*]] = arith.addf %[[IN0]], %[[HALF0]] : vector<64xf16>
// SPLIT:                vector.transfer_write %[[OUT0]], %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
// SPLIT:                arith.muli %{{.*}}, %[[C2]] : index
// SPLIT:                %[[IDX1:.*]] = arith.addi %{{.*}}, %[[C1]] : index
// SPLIT:                %[[IN1:.*]] = vector.transfer_read %[[COLLAPSED_IN]][%{{.*}}, %[[IDX1]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// SPLIT:                %[[OUT1:.*]] = arith.addf %[[IN1]], %[[HALF1]] : vector<64xf16>
// SPLIT:                vector.transfer_write %[[OUT1]], %{{.*}}[%{{.*}}, %[[IDX1]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
//
// No-split path (NOSPLIT) — inner loop runs over full 24 entries with divui/remui.
// NOSPLIT:      scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:        scf.for %[[D1:.*]] = %[[C0]] to %[[C24]] step %[[C1]]
// NOSPLIT:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// NOSPLIT:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:              %[[Q:.*]] = arith.divui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[R:.*]] = arith.remui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[Q]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// NOSPLIT:              %[[CMP:.*]] = arith.cmpi ne, %[[R]], %[[C0]] : index
// NOSPLIT:              %[[HALF0:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [0, 0, 1, 1
// NOSPLIT:              %[[HALF1:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [32, 32, 33, 33
// NOSPLIT:              %[[BIAS_SEL:.*]] = arith.select %[[CMP]], %[[HALF1]], %[[HALF0]] : vector<64xf16>
// NOSPLIT:              %[[IN:.*]] = vector.transfer_read %[[COLLAPSED_IN]][%{{.*}}, %[[D1]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// NOSPLIT:              %[[OUT:.*]] = arith.addf %[[IN]], %[[BIAS_SEL]] : vector<64xf16>
// NOSPLIT:              vector.transfer_write %[[OUT]], %{{.*}}[%{{.*}}, %[[D1]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
//
// COMMON:       tensor.expand_shape %{{.*}} {{.*}} : tensor<2x24x8x2x64xf16> into tensor<2x24x8x2x32x2xf16>

func.func @bias_add(%arg0: tensor<2x24x8x2x32x2xf16>,
                     %arg1: tensor<24x32xf16>) -> tensor<2x24x8x2x32x2xf16> {
  %init = tensor.empty() : tensor<2x24x8x2x32x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map6, #map5],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x24x8x2x32x2xf16>, tensor<24x32xf16>)
    outs(%init : tensor<2x24x8x2x32x2xf16>) {
  ^bb0(%in: f16, %in_136: f16, %out: f16):
    %0 = arith.addf %in, %in_136 : f16
    linalg.yield %0 : f16
  } -> tensor<2x24x8x2x32x2xf16>
  return %res : tensor<2x24x8x2x32x2xf16>
}

// ============================================================================
// Section 4: Parallel - simple GeLU-like (map6 bias + scalar tensor inputs)
//
// Maps: [map5, map6, map11, map11] -> map5 (all-parallel 6D)
// Body: GeLU polynomial + tanh
// Unique: map11 scalar tensor inputs (tensor<f16>), tanh in body.
// ============================================================================

// COMMON-LABEL: func.func @gelu
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// SPLIT-DAG:    %[[C48:.*]] = arith.constant 48 : index
// NOSPLIT-DAG:  %[[C96:.*]] = arith.constant 96 : index
// COMMON:       tensor.collapse_shape %{{.*}} {{.*}} : tensor<2x96x8x2x32x2xf16> into tensor<2x96x8x2x64xf16>
// COMMON:       %[[BIAS_EXP:.*]] = tensor.expand_shape %{{.*}} {{.*}} : tensor<96x32xf16> into tensor<48x2x32xf16>
// COMMON:       %[[BIAS_COL:.*]] = tensor.collapse_shape %[[BIAS_EXP]] {{.*}} : tensor<48x2x32xf16> into tensor<48x64xf16>
//
// Loop-split path (SPLIT) — inner loop bound 48, two unrolled iterations.
// SPLIT:        scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// SPLIT:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// SPLIT:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// SPLIT:              scf.for %[[IV:.*]] = %[[C0]] to %[[C48]] step %[[C1]] iter_args
// SPLIT:                %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<48x64xf16>, vector<64xf16>
// SPLIT:                vector.shuffle %[[BVEC]], %[[BVEC]] [0, 0, 1, 1
// SPLIT:                vector.shuffle %[[BVEC]], %[[BVEC]] [32, 32, 33, 33
// SPLIT:                %[[IDX0:.*]] = arith.muli %[[IV]], %[[C2]] : index
// SPLIT:                tensor.extract %{{.*}}[] : tensor<f16>
// SPLIT:                tensor.extract %{{.*}}[] : tensor<f16>
// SPLIT:                %[[VIN0:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : tensor<2x96x8x2x64xf16>, vector<64xf16>
// SPLIT:                %[[BIASED0:.*]] = arith.addf %[[VIN0]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[HALF0:.*]] = arith.mulf %[[BIASED0]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[TANH0:.*]] = math.tanh %{{.*}} : vector<64xf16>
// SPLIT:                %[[TANHP1:.*]] = arith.addf %[[TANH0]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[GELU0:.*]] = arith.mulf %[[HALF0]], %[[TANHP1]] : vector<64xf16>
// SPLIT:                vector.transfer_write %[[GELU0]], %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : vector<64xf16>, tensor<2x96x8x2x64xf16>
// SPLIT:                arith.muli %{{.*}}, %[[C2]] : index
// SPLIT:                %[[IDX1:.*]] = arith.addi %{{.*}}, %[[C1]] : index
// SPLIT:                %[[VIN1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[IDX1]]{{.*}} : tensor<2x96x8x2x64xf16>, vector<64xf16>
// SPLIT:                %[[BIASED1:.*]] = arith.addf %[[VIN1]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[HALF1:.*]] = arith.mulf %[[BIASED1]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[TANH1:.*]] = math.tanh %{{.*}} : vector<64xf16>
// SPLIT:                %[[TANHP1B:.*]] = arith.addf %[[TANH1]], %{{.*}} : vector<64xf16>
// SPLIT:                %[[GELU1:.*]] = arith.mulf %[[HALF1]], %[[TANHP1B]] : vector<64xf16>
// SPLIT:                vector.transfer_write %[[GELU1]], %{{.*}}[%{{.*}}, %[[IDX1]]{{.*}} : vector<64xf16>, tensor<2x96x8x2x64xf16>
//
// No-split path (NOSPLIT) — inner loop bound 96 with divui/remui.
// NOSPLIT:      scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:        scf.for %[[D1:.*]] = %[[C0]] to %[[C96]] step %[[C1]]
// NOSPLIT:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// NOSPLIT:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:              tensor.extract %{{.*}}[] : tensor<f16>
// NOSPLIT:              tensor.extract %{{.*}}[] : tensor<f16>
// NOSPLIT:              %[[Q:.*]] = arith.divui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[R:.*]] = arith.remui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[Q]], %[[C0]]]{{.*}} : tensor<48x64xf16>, vector<64xf16>
// NOSPLIT:              %[[CMP:.*]] = arith.cmpi ne, %[[R]], %[[C0]] : index
// NOSPLIT:              arith.select %[[CMP]], %{{.*}}, %{{.*}} : vector<64xf16>
// NOSPLIT:              %[[VIN:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[D1]]{{.*}} : tensor<2x96x8x2x64xf16>, vector<64xf16>
// NOSPLIT:              %[[BIASED:.*]] = arith.addf %[[VIN]], %{{.*}} : vector<64xf16>
// NOSPLIT:              %[[HALF:.*]] = arith.mulf %[[BIASED]], %{{.*}} : vector<64xf16>
// NOSPLIT:              %[[TANH:.*]] = math.tanh %{{.*}} : vector<64xf16>
// NOSPLIT:              %[[TANHP1:.*]] = arith.addf %[[TANH]], %{{.*}} : vector<64xf16>
// NOSPLIT:              %[[GELU:.*]] = arith.mulf %[[HALF]], %[[TANHP1]] : vector<64xf16>
// NOSPLIT:              vector.transfer_write %[[GELU]], %{{.*}}[%{{.*}}, %[[D1]]{{.*}} : vector<64xf16>, tensor<2x96x8x2x64xf16>
//
// COMMON:       tensor.expand_shape %{{.*}} {{.*}} : tensor<2x96x8x2x64xf16> into tensor<2x96x8x2x32x2xf16>

func.func @gelu(%arg0: tensor<2x96x8x2x32x2xf16>, %arg1: tensor<96x32xf16>, %arg2: tensor<f16>, %arg3: tensor<f16>) -> tensor<2x96x8x2x32x2xf16> {
  %cst_32 = arith.constant 5.000000e-01 : f16
  %cst_33 = arith.constant 1.000000e+00 : f16
  %init = tensor.empty() : tensor<2x96x8x2x32x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map6, #map11, #map11, #map5],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x96x8x2x32x2xf16>, tensor<96x32xf16>, tensor<f16>, tensor<f16>) outs(%init : tensor<2x96x8x2x32x2xf16>) {
  ^bb0(%in: f16, %in_138: f16, %in_139: f16, %in_140: f16, %out: f16):
    %118 = arith.addf %in, %in_138 : f16
    %119 = arith.mulf %118, %cst_32 : f16
    %120 = arith.mulf %118, %118 : f16
    %121 = arith.mulf %118, %120 : f16
    %122 = arith.mulf %121, %in_139 : f16
    %123 = arith.addf %118, %122 : f16
    %124 = arith.mulf %123, %in_140 : f16
    %125 = math.tanh %124 : f16
    %126 = arith.addf %125, %cst_33 : f16
    %127 = arith.mulf %119, %126 : f16
    linalg.yield %127 : f16
  } -> tensor<2x96x8x2x32x2xf16>
  return %res : tensor<2x96x8x2x32x2xf16>
}

// ============================================================================
// Section 5: Parallel - tricky normalization (map6 bias with (d1,d4) indexing)
//
// Maps: [map5, map10, map6, map6] -> map5 (all-parallel 6D)
// Body: layer-norm-like: rsqrt((mean/256 + eps)) * in * weight + bias
// Unique: map6 bias indexed by (d1,d4) requiring expand_shape+collapse of the
//         24x32 weight/bias tensors; rsqrt in body.
// ============================================================================

// COMMON-LABEL: func.func @layer_norm
// COMMON-SAME:  (%[[ARG0:.*]]: tensor<2x24x8x2x32x2xf16>, %[[ARG1:.*]]: tensor<2x8x2x2xf16>,
// COMMON-SAME:   %[[ARG2:.*]]: tensor<24x32xf16>, %[[ARG3:.*]]: tensor<24x32xf16>)
// COMMON-DAG:   %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:   %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:   %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:    %[[C12:.*]] = arith.constant 12 : index
// EAGER-DAG:    %[[C12:.*]] = arith.constant 12 : index
// NOSPLIT-DAG:  %[[C24:.*]] = arith.constant 24 : index
// COMMON:       tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<2x24x8x2x32x2xf16> into tensor<2x24x8x2x64xf16>
// COMMON:       %[[WGT_EXP:.*]] = tensor.expand_shape %[[ARG2]] {{.*}} : tensor<24x32xf16> into tensor<12x2x32xf16>
// COMMON:       %[[WGT_COL:.*]] = tensor.collapse_shape %[[WGT_EXP]] {{.*}} : tensor<12x2x32xf16> into tensor<12x64xf16>
// COMMON:       %[[BIAS_EXP:.*]] = tensor.expand_shape %[[ARG3]] {{.*}} : tensor<24x32xf16> into tensor<12x2x32xf16>
// COMMON:       %[[BIAS_COL:.*]] = tensor.collapse_shape %[[BIAS_EXP]] {{.*}} : tensor<12x2x32xf16> into tensor<12x64xf16>
//
// Default mode (CHECK) — scalar rsqrt on f16, loop-split with bound 12.
// CHECK:        scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:              scf.for %[[IV:.*]] = %[[C0]] to %[[C12]] step %[[C1]] iter_args
// CHECK:                %[[WVEC:.*]] = vector.transfer_read %[[WGT_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// CHECK:                %[[WHALF:.*]] = vector.shuffle %[[WVEC]], %[[WVEC]] [0, 0, 1, 1
// CHECK:                vector.shuffle %[[WVEC]], %[[WVEC]] [32, 32, 33, 33
// CHECK:                %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// CHECK:                %[[BHALF:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [0, 0, 1, 1
// CHECK:                vector.shuffle %[[BVEC]], %[[BVEC]] [32, 32, 33, 33
// CHECK:                %[[IDX0:.*]] = arith.muli %[[IV]], %[[C2]] : index
// CHECK:                %[[M0:.*]] = tensor.extract %[[ARG1]][{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// CHECK:                %[[M1:.*]] = tensor.extract %[[ARG1]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>
// CHECK:                %[[D0:.*]] = arith.divf %[[M0]], %{{.*}} : f16
// CHECK:                %[[D1:.*]] = arith.divf %[[M1]], %{{.*}} : f16
// CHECK:                arith.truncf %{{.*}} : f64 to f16
// CHECK:                %[[V0:.*]] = arith.addf %[[D0]], %{{.*}} : f16
// CHECK:                arith.addf %[[D1]], %{{.*}} : f16
// CHECK:                %[[RS0:.*]] = math.rsqrt %[[V0]] : f16
// CHECK:                %[[RS1:.*]] = math.rsqrt %{{.*}} : f16
// Data flow: input * interleaved_rsqrt * weight + bias → write
// CHECK:                %[[VIN:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// CHECK:                vector.broadcast %[[RS0]] : f16 to vector<32xf16>
// CHECK:                vector.broadcast %[[RS1]] : f16 to vector<32xf16>
// CHECK:                %[[RSQRT_VEC:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 32
// CHECK:                %[[SCALED:.*]] = arith.mulf %[[VIN]], %[[RSQRT_VEC]] : vector<64xf16>
// CHECK:                %[[WEIGHTED:.*]] = arith.mulf %[[SCALED]], %[[WHALF]] : vector<64xf16>
// CHECK:                %[[RESULT:.*]] = arith.addf %[[WEIGHTED]], %[[BHALF]] : vector<64xf16>
// CHECK:                vector.transfer_write %[[RESULT]], %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
//
// No-split mode (NOSPLIT) — scalar rsqrt on f16, divui/remui indexing, loop bound 24.
// NOSPLIT:      scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:        scf.for %[[D1:.*]] = %[[C0]] to %[[C24]] step %[[C1]]
// NOSPLIT:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// NOSPLIT:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// NOSPLIT:              %[[M0:.*]] = tensor.extract %[[ARG1]][{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// NOSPLIT:              %[[D0:.*]] = arith.divf %[[M0]], %{{.*}} : f16
// NOSPLIT:              arith.divf %{{.*}}, %{{.*}} : f16
// NOSPLIT:              %[[V0:.*]] = arith.addf %[[D0]], %{{.*}} : f16
// NOSPLIT:              %[[RS0:.*]] = math.rsqrt %[[V0]] : f16
// NOSPLIT:              %[[RS1:.*]] = math.rsqrt %{{.*}} : f16
// NOSPLIT:              %[[Q:.*]] = arith.divui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[R:.*]] = arith.remui %[[D1]], %[[C2]] : index
// NOSPLIT:              %[[WVEC:.*]] = vector.transfer_read %[[WGT_COL]][%[[Q]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// NOSPLIT:              %[[CMP:.*]] = arith.cmpi ne, %[[R]], %[[C0]] : index
// NOSPLIT:              %[[WH0:.*]] = vector.shuffle %[[WVEC]], %[[WVEC]] [0, 0, 1, 1
// NOSPLIT:              %[[WH1:.*]] = vector.shuffle %[[WVEC]], %[[WVEC]] [32, 32, 33, 33
// NOSPLIT:              %[[WSEL:.*]] = arith.select %[[CMP]], %[[WH1]], %[[WH0]] : vector<64xf16>
// NOSPLIT:              vector.transfer_read %{{.*}}[%{{.*}}, %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// NOSPLIT:              %[[BSEL:.*]] = arith.select %{{.*}} : vector<64xf16>
// Data flow: input * interleaved_rsqrt * weight_selected + bias_selected → write
// NOSPLIT:              %[[VIN:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[D1]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// NOSPLIT:              vector.broadcast %[[RS0]] : f16 to vector<32xf16>
// NOSPLIT:              vector.broadcast %[[RS1]] : f16 to vector<32xf16>
// NOSPLIT:              %[[RSQRT_VEC:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 32
// NOSPLIT:              %[[SCALED:.*]] = arith.mulf %[[VIN]], %[[RSQRT_VEC]] : vector<64xf16>
// NOSPLIT:              %[[WEIGHTED:.*]] = arith.mulf %[[SCALED]], %[[WSEL]] : vector<64xf16>
// NOSPLIT:              %[[RESULT:.*]] = arith.addf %[[WEIGHTED]], %[[BSEL]] : vector<64xf16>
// NOSPLIT:              vector.transfer_write %[[RESULT]], %{{.*}}[%{{.*}}, %[[D1]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
//
// Eager mode (EAGER) — vector rsqrt on vector<64xf16>, loop-split with bound 12.
// EAGER:        scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// EAGER:          scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// EAGER:            scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
// EAGER:              scf.for %[[IV:.*]] = %[[C0]] to %[[C12]] step %[[C1]]
// EAGER:                %[[WVEC:.*]] = vector.transfer_read %[[WGT_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// EAGER:                %[[WHALF:.*]] = vector.shuffle %[[WVEC]], %[[WVEC]] [0, 0, 1, 1
// EAGER:                vector.shuffle %[[WVEC]], %[[WVEC]] [32, 32, 33, 33
// EAGER:                %[[BVEC:.*]] = vector.transfer_read %[[BIAS_COL]][%[[IV]], %[[C0]]]{{.*}} : tensor<12x64xf16>, vector<64xf16>
// EAGER:                %[[BHALF:.*]] = vector.shuffle %[[BVEC]], %[[BVEC]] [0, 0, 1, 1
// EAGER:                vector.shuffle %[[BVEC]], %[[BVEC]] [32, 32, 33, 33
// EAGER:                %[[IDX0:.*]] = arith.muli %[[IV]], %[[C2]] : index
// EAGER:                %[[M0:.*]] = tensor.extract %[[ARG1]][{{.*}}, %[[C0]]] : tensor<2x8x2x2xf16>
// EAGER:                %[[M1:.*]] = tensor.extract %[[ARG1]][{{.*}}, %[[C1]]] : tensor<2x8x2x2xf16>
// Data flow: broadcast means → interleave → vector divf → vector rsqrt
// EAGER:                %[[BM0:.*]] = vector.broadcast %[[M0]] : f16 to vector<32xf16>
// EAGER:                %[[BM1:.*]] = vector.broadcast %[[M1]] : f16 to vector<32xf16>
// EAGER:                %[[MEAN_VEC:.*]] = vector.shuffle %[[BM0]], %[[BM1]] [0, 32
// EAGER:                %[[DIVD:.*]] = arith.divf %[[MEAN_VEC]], %{{.*}} : vector<64xf16>
// EAGER:                arith.truncf %{{.*}} : f64 to f16
// EAGER:                %[[VRSQRT:.*]] = math.rsqrt %{{.*}} : vector<64xf16>
// Data flow: input * vector_rsqrt * weight + bias → write
// EAGER:                %[[VIN:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : tensor<2x24x8x2x64xf16>, vector<64xf16>
// EAGER:                %[[SCALED:.*]] = arith.mulf %[[VIN]], %[[VRSQRT]] : vector<64xf16>
// EAGER:                %[[WEIGHTED:.*]] = arith.mulf %[[SCALED]], %[[WHALF]] : vector<64xf16>
// EAGER:                %[[RESULT:.*]] = arith.addf %[[WEIGHTED]], %[[BHALF]] : vector<64xf16>
// EAGER:                vector.transfer_write %[[RESULT]], %{{.*}}[%{{.*}}, %[[IDX0]]{{.*}} : vector<64xf16>, tensor<2x24x8x2x64xf16>
//
// COMMON:       tensor.expand_shape %{{.*}} {{.*}} : tensor<2x24x8x2x64xf16> into tensor<2x24x8x2x32x2xf16>

func.func @layer_norm(%arg0: tensor<2x24x8x2x32x2xf16>, %arg1: tensor<2x8x2x2xf16>, %arg2: tensor<24x32xf16>, %arg3: tensor<24x32xf16>) -> tensor<2x24x8x2x32x2xf16> {
  %cst29 = arith.constant 1.000000e-05 : f64
  %cst31 = arith.constant 2.560000e+02 : f16
  %init = tensor.empty() : tensor<2x24x8x2x32x2xf16>
  %res = linalg.generic {
    indexing_maps = [#map5, #map10, #map6, #map6, #map5],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x24x8x2x32x2xf16>, tensor<2x8x2x2xf16>, tensor<24x32xf16>, tensor<24x32xf16>) outs(%init : tensor<2x24x8x2x32x2xf16>) {
  ^bb0(%in: f16, %in_136: f16, %in_137: f16, %in_138: f16, %out: f16):
    %0 = arith.divf %in_136, %cst31 : f16
    %1 = arith.truncf %cst29 : f64 to f16
    %2 = arith.addf %0, %1 : f16
    %3 = math.rsqrt %2 : f16
    %4 = arith.mulf %in, %3 : f16
    %5 = arith.mulf %4, %in_137 : f16
    %6 = arith.addf %5, %in_138 : f16
    linalg.yield %6 : f16
  } -> tensor<2x24x8x2x32x2xf16>
  return %res : tensor<2x24x8x2x32x2xf16>
}

// ============================================================================
// Section 6: Dynamic shapes (regression - bail out, do not crash)
//
// Previously, VectorizationStrategy::analyze() called
// RankedTensorType::getNumElements() unconditionally, which asserts/crashes
// when any dimension is dynamic. The pass must bail out conservatively and
// leave the linalg.generic unchanged.
// ============================================================================

// COMMON-LABEL: func.func @dynamic_matmul_packed
// Pass must bail out on dynamic shapes — op is left unchanged.
// CHECK: linalg.generic

func.func @dynamic_matmul_packed(
    %arg0: tensor<?x?x32x2xf16>,
    %arg1: tensor<?x?x32x2xf16>) -> tensor<?x?x32x2xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x32x2xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x32x2xf16>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?x32x2xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x32x2xf16>, tensor<?x?x32x2xf16>)
    outs(%empty : tensor<?x?x32x2xf16>) {
  ^bb0(%in0: f16, %in1: f16, %out: f16):
    %add = arith.addf %in0, %in1 : f16
    linalg.yield %add : f16
  } -> tensor<?x?x32x2xf16>
  return %result : tensor<?x?x32x2xf16>
}
