//===- hexagonmem_workspace_resident.mlir - resident workspace lowering ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A resident workspace alloc does not go through the per-launch allocator: the
// lowering calls the runtime's workspace-resident entry with the compile-time
// key the marking pass put on the allocation. Nothing is copied -- the kernel
// owns the contents and refills the buffer every launch.
//
// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @hexagon_runtime_workspace_resident_dsp(i64, i32) -> !llvm.ptr

module {
  // CHECK-LABEL: func.func @resident_workspace
  func.func @resident_workspace() {
    // The key is a compile-time constant (the runtime's residency map key), not
    // an address.
    // CHECK: %[[KEY:.*]] = arith.constant -5511315095222747136 : i64
    // CHECK: %[[BYTES:.*]] = llvm.mlir.constant(8192 : i32) : i32
    // CHECK: llvm.call @hexagon_runtime_workspace_resident_dsp(%[[KEY]], %[[BYTES]]) : (i64, i32) -> !llvm.ptr
    // The call's result becomes the buffer's memref descriptor.
    // CHECK: llvm.insertvalue
    %w = hexagonmem.alloc() {alignment = 128 : i64, hmx.workspace_resident = {bytes = 8192 : i64, key = -5511315095222747136 : i64}} : memref<2x2x16x32x2xf16, 1>
    return
  }
}
