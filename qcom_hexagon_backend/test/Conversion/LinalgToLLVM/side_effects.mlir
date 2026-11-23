//===- side_effects.mlir --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -linalg-to-llvm="enable-hexkl=true" | FileCheck %s
func.func @side_effects(%arg0: tensor<32xf32>, %arg1: memref<?xf32>, %arg2: tensor<32xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg2 : tensor<32xf32>, tensor<32xi32>) {
  ^bb0(%in: f32, %idx: i32):
    %1 = arith.index_cast %idx : i32 to index
    memref.store %in, %arg1[%1] : memref<?xf32>
    linalg.yield
  }
  return
}

// CHECK:   llvm.func @side_effects
// CHECK:       %[[IDX:.*]] = llvm.sext %{{.*}} : i32 to i64
// CHECK:       %[[PTR:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:       llvm.store %{{.*}}, %[[PTR]] : f32, !llvm.ptr
// CHECK:       llvm.return
