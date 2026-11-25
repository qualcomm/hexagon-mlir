//===- vtcm_tiling_scf_for.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// This op is a pure elementwise add, which VTCMTilingPass now skips staging
// into VTCM by default (see [VTCMTiling] Skip VTCM staging for pure
// elementwise generics). Force tiling via tile-size-override so this test
// keeps exercising VTCM staging inside an scf.for with return-alloc-from-loop
// bufferization, which is what this test actually verifies.
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(vtcm-tiling{tile-size-override=32}),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' | FileCheck %s

func.func @add_kernel() -> tensor<32xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %result = scf.for %I = %c0_i32 to %c2_i32 step %c1_i32
                          iter_args(%cF = %fill) -> (tensor<32xf32>)  : i32 {
    %genericRes = linalg.generic
             {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]} ins(%cF : tensor<32xf32>) outs(%0: tensor<32xf32>) {
    ^bb0(%in_2: f32, %out_3: f32):
      %5 = arith.addf %in_2, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<32xf32>
    scf.yield %genericRes : tensor<32xf32>
  }
  return %result : tensor<32xf32>
}

// CHECK: %[[ALLOCDDR:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// Output is write-only (%out_3 never read in body), so no DDR->VTCM copy.
// CHECK: %[[ALLOC_VTCM_OUT:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
// CHECK-NOT: memref.copy %[[ALLOCDDR]], %[[ALLOC_VTCM_OUT]]
// CHECK: linalg.fill ins({{.*}} : f32) outs(%[[ALLOCDDR]] : memref<32xf32>)
// CHECK: %[[FOR:.+]] = scf.for {{.*}} iter_args(%arg1 = %[[ALLOCDDR]]) -> (memref<32xf32>)  : i32 {
// CHECK:       %[[ALLOC_VTCM_INP:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
// CHECK-NEXT:  memref.copy %arg1, %[[ALLOC_VTCM_INP]] : memref<32xf32> to memref<32xf32, 1>
// CHECK:       linalg.generic {{.*}} ins(%[[ALLOC_VTCM_INP]] : memref<32xf32, 1>) outs(%[[ALLOC_VTCM_OUT]] : memref<32xf32, 1>) {
// CHECK:       %[[ALLOC_RES:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// CHECK-NEXT:  memref.copy %[[ALLOC_VTCM_OUT]], %[[ALLOC_RES]] : memref<32xf32, 1> to memref<32xf32>
// CHECK-NEXT:  scf.yield %[[ALLOC_RES]] : memref<32xf32>
