//===- convert-static-alloc-to-alloca-budget.mlir -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -static-alloc-to-alloca='max-total-alloca-size-bytes=3' | FileCheck %s
// RUN: linalg-hexagon-opt %s -static-alloc-to-alloca='max-total-alloca-size-bytes=3' 2>&1 | FileCheck %s --check-prefix=WARN

// Greedy budget: three allocs with effective sizes 1 B, 2 B, 4 B and a
// cumulative budget of 3 B. The two cheapest (1xi8, 1xi16) are promoted;
// the largest (1xi32) is rejected and kept on the heap.

// WARN: warning: StaticAllocToAlloca: 1 alloc(s) not promoted to alloca
// WARN-SAME: adding each would push the peak live budget over 3 B

// CHECK-LABEL: func.func @greedy_budget_partial_conversion
func.func @greedy_budget_partial_conversion() {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %{{.*}} = memref.alloca() : memref<1xi8>
  // CHECK-DAG: %{{.*}} = memref.alloca() : memref<1xi16>
  %a = memref.alloc() : memref<1xi8>
  %b = memref.alloc() : memref<1xi16>
  // CHECK:     %[[C:.*]] = memref.alloc() : memref<1xi32>
  // CHECK-NOT: memref.alloca() : memref<1xi32>
  %c = memref.alloc() : memref<1xi32>
  %v8  = arith.constant 1 : i8
  %v16 = arith.constant 2 : i16
  %v32 = arith.constant 3 : i32
  memref.store %v8,  %a[%c0] : memref<1xi8>
  memref.store %v16, %b[%c0] : memref<1xi16>
  memref.store %v32, %c[%c0] : memref<1xi32>
  memref.dealloc %a : memref<1xi8>
  memref.dealloc %b : memref<1xi16>
  // CHECK: memref.dealloc %[[C]] : memref<1xi32>
  memref.dealloc %c : memref<1xi32>
  return
}
