//===- pow_large_vector_hexagon_routines.mlir -----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' | FileCheck %s

// CHECK-LABEL: llvm.func @_hexagon_runtime_pow__vf(vector<32xi32>, vector<32xi32>) -> vector<32xi32>
// CHECK-LABEL: llvm.func @pow_large
// CHECK-NOT: llvm.bitcast {{.*}} : vector<128xf32> to vector<32xi32>
// CHECK: [[BASE0:%.*]] = llvm.shufflevector %arg0, {{%.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<128xf32>
// CHECK-NEXT: [[EXP0:%.*]] = llvm.shufflevector %arg1, {{%.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<128xf32>
// CHECK-NEXT: [[BASE0I:%.*]] = llvm.bitcast [[BASE0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP0I:%.*]] = llvm.bitcast [[EXP0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[CALL0:%.*]] = llvm.call @_hexagon_runtime_pow__vf([[BASE0I]], [[EXP0I]])
// CHECK-NOT: llvm.bitcast {{.*}} : vector<128xf32> to vector<32xi32>
// CHECK: [[BASE1:%.*]] = llvm.shufflevector %arg0, {{%.*}} [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<128xf32>
// CHECK-NEXT: [[EXP1:%.*]] = llvm.shufflevector %arg1, {{%.*}} [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<128xf32>
// CHECK-NEXT: [[BASE1I:%.*]] = llvm.bitcast [[BASE1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP1I:%.*]] = llvm.bitcast [[EXP1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: llvm.call @_hexagon_runtime_pow__vf([[BASE1I]], [[EXP1I]])
// CHECK-NOT: llvm.bitcast {{.*}} : vector<128xf32> to vector<32xi32>
// CHECK: [[BASE2:%.*]] = llvm.shufflevector %arg0, {{%.*}} [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95] : vector<128xf32>
// CHECK-NEXT: [[EXP2:%.*]] = llvm.shufflevector %arg1, {{%.*}} [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95] : vector<128xf32>
// CHECK-NEXT: [[BASE2I:%.*]] = llvm.bitcast [[BASE2]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP2I:%.*]] = llvm.bitcast [[EXP2]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: llvm.call @_hexagon_runtime_pow__vf([[BASE2I]], [[EXP2I]])
// CHECK-NOT: llvm.bitcast {{.*}} : vector<128xf32> to vector<32xi32>
// CHECK: [[BASE3:%.*]] = llvm.shufflevector %arg0, {{%.*}} [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf32>
// CHECK-NEXT: [[EXP3:%.*]] = llvm.shufflevector %arg1, {{%.*}} [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf32>
// CHECK-NEXT: [[BASE3I:%.*]] = llvm.bitcast [[BASE3]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP3I:%.*]] = llvm.bitcast [[EXP3]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: llvm.call @_hexagon_runtime_pow__vf([[BASE3I]], [[EXP3I]])
// CHECK-NOT: llvm.bitcast {{.*}} : vector<128xf32> to vector<32xi32>

llvm.func @pow_large(%arg0: vector<128xf32>, %arg1: vector<128xf32>)
    -> vector<128xf32> {
  %0 = llvm.intr.pow(%arg0, %arg1) {fastmathFlags = #llvm.fastmath<fast>}
      : (vector<128xf32>, vector<128xf32>) -> vector<128xf32>
  llvm.return %0 : vector<128xf32>
}
