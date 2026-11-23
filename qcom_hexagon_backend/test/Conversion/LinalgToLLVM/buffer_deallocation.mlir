//===- buffer_deallocation.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -linalg-to-llvm="dealloc-after-last-use=false" | FileCheck %s

module {
  // Small alloc: 8xf32 = 32 B data; effectiveBytes = alignTo(32, 64) = 64 B.
  // 64 B <= 64 B per-alloc cap AND <= 4096 B cumulative budget → promoted to alloca.
  func.func @buffer_alloc_small(%arg0: memref<8xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    memref.copy %alloc, %arg0 : memref<8xf32> to memref<8xf32>
    return
  }

  // Large alloc: 4096xf32 = 16384 B effective (alignTo(16384, 64) = 16384 B).
  // 16384 B > 64 B per-alloc cap AND > 4096 B cumulative budget → stays as malloc.
  func.func @buffer_alloc_large(%arg0: memref<4096xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
    memref.copy %alloc, %arg0 : memref<4096xf32> to memref<4096xf32>
    return
  }
}

// CHECK-LABEL: @buffer_alloc_small
// CHECK:      %{{.*}} = llvm.alloca %{{.*}} x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
// CHECK-NOT:  llvm.call @malloc
// CHECK-NOT:  llvm.call @free

// CHECK-LABEL: @buffer_alloc_large
// CHECK:      %[[ALLOC:.+]] = llvm.call @malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:      llvm.call @free(%[[ALLOC]]) : (!llvm.ptr) -> ()
