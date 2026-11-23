//===- convert-static-alloc-to-alloca.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -static-alloc-to-alloca | FileCheck %s

// CHECK-LABEL: func.func @basic_scalar_converted
func.func @basic_scalar_converted() -> f32 {
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<f32>
  // CHECK-NOT: memref.alloc() : memref<f32>
  %alloc = memref.alloc() : memref<f32>
  %cst = arith.constant 1.0 : f32
  // CHECK: memref.store %{{.*}}, %[[ALLOCA]]
  memref.store %cst, %alloc[] : memref<f32>
  // CHECK: %{{.*}} = memref.load %[[ALLOCA]]
  %val = memref.load %alloc[] : memref<f32>
  // CHECK-NOT: memref.dealloc {{.*}} : memref<f32>
  memref.dealloc %alloc : memref<f32>
  return %val : f32
}

// CHECK-LABEL: func.func @loop_body_alloc_hoisted
func.func @loop_body_alloc_hoisted() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // alloca is hoisted to entry block, appearing before the loop
  // CHECK:     %[[TAG:.*]] = memref.alloca() : memref<1xi32>
  // CHECK:     scf.for
  // CHECK-NOT: memref.alloc() : memref<1xi32>
  // CHECK-NOT: memref.dealloc {{.*}} : memref<1xi32>
  scf.for %i = %c0 to %c10 step %c1 {
    %tag = memref.alloc() : memref<1xi32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %tag[%c0] : memref<1xi32>
    memref.dealloc %tag : memref<1xi32>
  }
  return
}

// CHECK-LABEL: func.func @alignment_preserved
func.func @alignment_preserved() {
  // CHECK: %{{.*}} = memref.alloca() {alignment = 64 : i64} : memref<f16>
  // CHECK-NOT: memref.alloc() {alignment = 64 : i64} : memref<f16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f16>
  %cst = arith.constant 1.0 : f16
  memref.store %cst, %alloc[] : memref<f16>
  // CHECK-NOT: memref.dealloc {{.*}} : memref<f16>
  memref.dealloc %alloc : memref<f16>
  return
}

// CHECK-LABEL: func.func @multiple_types_all_converted
func.func @multiple_types_all_converted() {
  // CHECK-DAG: %{{.*}} = memref.alloca() {alignment = 64 : i64} : memref<f16>
  // CHECK-DAG: %{{.*}} = memref.alloca() {alignment = 64 : i64} : memref<f32>
  // CHECK-DAG: %{{.*}} = memref.alloca() : memref<1xi32>
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() {alignment = 64 : i64} : memref<f16>
  %b = memref.alloc() {alignment = 64 : i64} : memref<f32>
  %c = memref.alloc() : memref<1xi32>
  %cst_f16 = arith.constant 1.0 : f16
  %cst_f32 = arith.constant 2.0 : f32
  %c42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  memref.store %cst_f16, %a[] : memref<f16>
  memref.store %cst_f32, %b[] : memref<f32>
  memref.store %c42, %c[%c0] : memref<1xi32>
  memref.dealloc %a : memref<f16>
  memref.dealloc %b : memref<f32>
  memref.dealloc %c : memref<1xi32>
  return
}

// CHECK-LABEL: func.func @above_total_budget_not_converted
func.func @above_total_budget_not_converted() {
  // effectiveBytes = alignTo(16384, 64) = 16384 > default budget 2048 B
  // CHECK:     %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  // CHECK-NOT: memref.alloca() {{.*}} : memref<4096xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  // CHECK: memref.dealloc {{.*}} : memref<4096xf32>
  memref.dealloc %alloc : memref<4096xf32>
  return
}

// CHECK-LABEL: func.func @vtcm_alloc_not_converted
func.func @vtcm_alloc_not_converted() {
  // CHECK:     %{{.*}} = memref.alloc() : memref<f32, 1>
  // CHECK-NOT: memref.alloca() : memref<f32, 1>
  %alloc = memref.alloc() : memref<f32, 1>
  // CHECK: memref.dealloc {{.*}} : memref<f32, 1>
  memref.dealloc %alloc : memref<f32, 1>
  return
}

// CHECK-LABEL: func.func @dynamic_alloc_not_converted
func.func @dynamic_alloc_not_converted(%n: index) {
  // CHECK:     %{{.*}} = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK-NOT: memref.alloca() : memref<?xf32>
  %alloc = memref.alloc(%n) : memref<?xf32>
  // CHECK: memref.dealloc {{.*}} : memref<?xf32>
  memref.dealloc %alloc : memref<?xf32>
  return
}

// CHECK-LABEL: func.func @independent_budget_func_a
func.func @independent_budget_func_a() {
  // CHECK: %{{.*}} = memref.alloca() : memref<f32>
  %a = memref.alloc() : memref<f32>
  memref.dealloc %a : memref<f32>
  return
}

// CHECK-LABEL: func.func @independent_budget_func_b
func.func @independent_budget_func_b() {
  // CHECK: %{{.*}} = memref.alloca() : memref<f32>
  %b = memref.alloc() : memref<f32>
  memref.dealloc %b : memref<f32>
  return
}

// ---------------------------------------------------------------------------
// Liveness-aware peak budget tests.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @four_simultaneous_large_allocs_partial_reject
func.func @four_simultaneous_large_allocs_partial_reject() {
  %c0 = arith.constant 0 : index
  // Greedy accept (sorted ascending, all 1024 B each):
  //   1st candidate: peak = 1024 B <= 2048 B → promoted.
  //   2nd candidate: peak = 2048 B <= 2048 B → promoted.
  //   3rd candidate: peak = 3072 B > 2048 B  → rejected, stop.
  // Result: first two promoted to alloca, last two stay on heap.
  // CHECK:     %{{.*}} = memref.alloca() : memref<256xi32>
  // CHECK:     %{{.*}} = memref.alloca() : memref<256xi32>
  // CHECK:     %[[C:.*]] = memref.alloc() : memref<256xi32>
  // CHECK:     %[[D:.*]] = memref.alloc() : memref<256xi32>
  %a = memref.alloc() : memref<256xi32>
  %b = memref.alloc() : memref<256xi32>
  %c = memref.alloc() : memref<256xi32>
  %d = memref.alloc() : memref<256xi32>
  %c1 = arith.constant 1 : i32
  memref.store %c1, %a[%c0] : memref<256xi32>
  memref.store %c1, %b[%c0] : memref<256xi32>
  memref.store %c1, %c[%c0] : memref<256xi32>
  memref.store %c1, %d[%c0] : memref<256xi32>
  // CHECK: memref.dealloc %[[C]] : memref<256xi32>
  // CHECK: memref.dealloc %[[D]] : memref<256xi32>
  memref.dealloc %a : memref<256xi32>
  memref.dealloc %b : memref<256xi32>
  memref.dealloc %c : memref<256xi32>
  memref.dealloc %d : memref<256xi32>
  return
}

// CHECK-LABEL: func.func @sequential_large_allocs_coalesced
func.func @sequential_large_allocs_coalesced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  // Both allocs are non-overlapping so they coalesce into one alloca slot.
  // peak = 1024 B <= 2048 B → both promoted, sharing one alloca.
  // CHECK-COUNT-1: memref.alloca() : memref<256xi32>
  // CHECK-NOT:     memref.alloc() : memref<256xi32>
  %first = memref.alloc() : memref<256xi32>
  memref.store %c1, %first[%c0] : memref<256xi32>
  memref.dealloc %first : memref<256xi32>
  %second = memref.alloc() : memref<256xi32>
  memref.store %c1, %second[%c0] : memref<256xi32>
  memref.dealloc %second : memref<256xi32>
  // CHECK-NOT: memref.dealloc {{.*}} : memref<256xi32>
  return
}
