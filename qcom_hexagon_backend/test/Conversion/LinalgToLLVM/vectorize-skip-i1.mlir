//===- vectorize-skip-i1.mlir ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt -split-input-file %s -hexagon-vectorization | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// i1 results from body ops (arith.cmpf, arith.select) must not pin
// dataTileSize to 1 byte / 128 lanes; this f32 generic must still vectorize at
// the f32 tile size (32 lanes for a 128-byte HVX vector).

// CHECK-LABEL: func.func @elemwise_f32_with_cmpf_select
// CHECK:         vector.transfer_read {{.*}} : tensor<32xf32>, vector<32xf32>
// CHECK:         arith.cmpf {{.*}} : vector<32xf32>
// CHECK:         arith.select {{.*}} vector<32xi1>, vector<32xf32>
// CHECK:         vector.transfer_write {{.*}} : vector<32xf32>, tensor<32xf32>
func.func @elemwise_f32_with_cmpf_select(%a: tensor<32xf32>, %b: tensor<32xf32>) -> tensor<32xf32> {
  %cst = arith.constant 0.0 : f32
  %result = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<32xf32>) outs(%b : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ogt, %in, %cst : f32
      %sel = arith.select %cmp, %in, %cst : f32
      linalg.yield %sel : f32
  } -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// Same skip behavior for integer compare/and chains producing i1.

// CHECK-LABEL: func.func @elemwise_i32_with_cmpi_andi
// CHECK:         vector.transfer_read {{.*}} : tensor<32xi32>, vector<32xi32>
// CHECK:         arith.cmpi {{.*}} : vector<32xi32>
// CHECK:         arith.andi {{.*}} : vector<32xi1>
// CHECK:         arith.select {{.*}} vector<32xi1>, vector<32xi32>
// CHECK:         vector.transfer_write {{.*}} : vector<32xi32>, tensor<32xi32>
func.func @elemwise_i32_with_cmpi_andi(%a: tensor<32xi32>, %b: tensor<32xi32>) -> tensor<32xi32> {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %result = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<32xi32>) outs(%b : tensor<32xi32>) {
    ^bb0(%in: i32, %out: i32):
      %cmp1 = arith.cmpi sgt, %in, %c0 : i32
      %cmp2 = arith.cmpi slt, %in, %c10 : i32
      %and = arith.andi %cmp1, %cmp2 : i1
      %sel = arith.select %and, %in, %c0 : i32
      linalg.yield %sel : i32
  } -> tensor<32xi32>
  return %result : tensor<32xi32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// An i1 tensor operand (used as a mask) must be skipped when computing the
// smallest operand size; the f32 data operands drive the tile size.

// CHECK-LABEL: func.func @elemwise_f32_with_i1_mask_operand
// CHECK:         vector.transfer_read {{.*}} : tensor<32xi1>, vector<32xi1>
// CHECK:         vector.transfer_read {{.*}} : tensor<32xf32>, vector<32xf32>
// CHECK:         arith.select {{.*}} vector<32xi1>, vector<32xf32>
// CHECK:         vector.transfer_write {{.*}} : vector<32xf32>, tensor<32xf32>
func.func @elemwise_f32_with_i1_mask_operand(%mask: tensor<32xi1>, %a: tensor<32xf32>, %b: tensor<32xf32>) -> tensor<32xf32> {
  %cst = arith.constant 0.0 : f32
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%mask, %a : tensor<32xi1>, tensor<32xf32>) outs(%b : tensor<32xf32>) {
    ^bb0(%m: i1, %in: f32, %out: f32):
      %sel = arith.select %m, %in, %cst : f32
      linalg.yield %sel : f32
  } -> tensor<32xf32>
  return %result : tensor<32xf32>
}
