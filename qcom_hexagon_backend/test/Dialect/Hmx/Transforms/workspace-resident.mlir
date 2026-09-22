//===- workspace-resident.mlir - per-launch VTCM workspace becomes resident ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Every HMX kernel allocates and frees its VTCM workspace on every launch. The
// workspace has no compile-time contents (the kernel refills it), so all it
// needs is a stable buffer: this pass keys each static VTCM allocation and tags
// it, and its lowering replaces the per-launch alloc/free pair with one call to
// the runtime's resident entry.
//
// The key is a compile-time constant (function symbol hash + allocation index,
// top bit set), never an address: it has to be the same on every launch of the
// same kernel.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// Each static VTCM allocation of an HMX kernel is tagged with the same byte
// count the type encodes and a distinct key; the per-launch deallocation is
// dropped, because the buffer's lifetime is now the process, not the launch.
// CHECK-LABEL: func.func @workspace
func.func @workspace() {
  // CHECK: %[[A:.*]] = memref.alloc() {hmx.workspace_resident = {bytes = 8192 : i64, key = -{{[0-9]+}} : i64}} : memref<2x2x16x32x2xf16, 1>
  %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  // CHECK: %[[B:.*]] = memref.alloc() {hmx.workspace_resident = {bytes = 8192 : i64, key = -{{[0-9]+}} : i64}} : memref<2x2x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  // CHECK: %[[C:.*]] = memref.alloc() {hmx.workspace_resident = {bytes = 8192 : i64, key = -{{[0-9]+}} : i64}} : memref<2x2x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  // CHECK-NOT: memref.dealloc
  memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %w : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
  return
}

// -----

// A function the HMX path never touched keeps its allocations and its
// deallocations exactly as they were: residency is only for the HMX workspace.
// CHECK-LABEL: func.func @no_hmx
func.func @no_hmx() {
  // CHECK: memref.alloc() : memref<64xi8, 1>
  %a = memref.alloc() : memref<64xi8, 1>
  // CHECK-NOT: hmx.workspace_resident
  memref.dealloc %a : memref<64xi8, 1>
  return
}

// -----

// The weight path already pinned its buffer as a `hexagonmem.alloc`; the
// workspace pass leaves it to the weight residency (a different key space).
// CHECK-LABEL: func.func @weight_resident_untouched
func.func @weight_resident_untouched(
    %a: memref<2x2x16x32x2xf16, 1>, %r: memref<2x2x16x32x2xf16, 1>) {
  // CHECK: %[[W:.*]] = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @__w}} : memref<2x2x16x32x2xf16, 1>
  %w = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @__w}} : memref<2x2x16x32x2xf16, 1>
  // CHECK-NOT: hmx.workspace_resident
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}
