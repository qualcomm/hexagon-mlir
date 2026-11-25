//===- lower_core_ptr_ops_to_llvm.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(finalize-memref-and-core-ptr-to-llvm-conversion{enable-collapse-address-space=false})' | FileCheck %s
// Mirrors the tptr.* test in TptrToLinalg/lower_tptr_ops_to_llvm.mlir.

// Test ptr.to_ptr -> ptr.ptr_add -> ptr.from_ptr roundtrip: converts a memref
// to a raw pointer, advances it by a byte offset, and reconstructs the memref.
module {
func.func @ptradd(%base : memref<1xi32>, %offset : i32) -> memref<1xi32> {
  %ms = memref.memory_space_cast %base : memref<1xi32> to memref<1xi32, #tptr.default_memory_space>
  %0 = ptr.to_ptr %ms : memref<1xi32, #tptr.default_memory_space> -> <#tptr.default_memory_space>
  %1 = ptr.ptr_add %0, %offset : !ptr.ptr<#tptr.default_memory_space>, i32
  %2 = ptr.from_ptr %1 : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
  %3 = memref.memory_space_cast %2 : memref<1xi32, #tptr.default_memory_space> to memref<1xi32>
  return %3 : memref<1xi32>
}
}

// The conversion marks the ptr dialect illegal, so successful completion of
// the pass verifies that all three ptr operations were lowered.
// CHECK-LABEL: func.func @ptradd
// CHECK: llvm.getelementptr {{.*}}[%arg1] : (!llvm.ptr, i32) -> !llvm.ptr, i8




// -----

// Handle memrefs of ptrs
func.func @ptr_memref_roundtrip(%base : memref<1xi32>, %i : index) -> memref<1xi32> {
  %ms = memref.memory_space_cast %base : memref<1xi32> to memref<1xi32, #tptr.default_memory_space>
  %p = ptr.to_ptr %ms : memref<1xi32, #tptr.default_memory_space> -> <#tptr.default_memory_space>
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<4x!ptr.ptr<#tptr.default_memory_space>>
  memref.store %p, %alloc[%i] : memref<4x!ptr.ptr<#tptr.default_memory_space>>
  %loaded = memref.load %alloc[%i] : memref<4x!ptr.ptr<#tptr.default_memory_space>>
  memref.dealloc %alloc : memref<4x!ptr.ptr<#tptr.default_memory_space>>
  %2 = ptr.from_ptr %loaded : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
  %3 = memref.memory_space_cast %2 : memref<1xi32, #tptr.default_memory_space> to memref<1xi32>
  return %3 : memref<1xi32>
}

// CHECK-LABEL:   func.func @ptr_memref_roundtrip
// CHECK-NOT:       memref.alloc
// CHECK-NOT:       memref.store
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.dealloc
// memref.alloc: lowered via the standard malloc-based allocation sequence.
// CHECK:           llvm.call @malloc
// memref.store: the `!llvm.ptr` value is stored directly (no ptrtoint needed).
// CHECK:           llvm.store %{{.*}}, %{{.*}} : !llvm.ptr, !llvm.ptr
// memref.load: the `!llvm.ptr` value is loaded directly (no inttoptr needed).
// CHECK:           %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
// memref.dealloc: lowered via the standard free call.
// CHECK:           llvm.call @free
