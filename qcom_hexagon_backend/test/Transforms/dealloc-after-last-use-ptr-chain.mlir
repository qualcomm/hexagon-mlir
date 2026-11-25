//===- dealloc-after-last-use-ptr-chain.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Tests for two dealloc-placement fixes in HexagonDeallocAfterLastUsePass:
//
//  1. extract_aligned_pointer_as_index / llvm.call pointer chain:
//     A buffer's last SSA (memref) use is extract_aligned_pointer_as_index,
//     but the extracted pointer flows through arith.index_cast + llvm.inttoptr
//     into an llvm.call that still reads the memory.  The dealloc must land
//     AFTER the call, not after the extract.
//
//  2. scf.if branch yield:
//     A buffer allocated inside an scf.if branch and yielded out via scf.yield
//     must NOT be deallocated inside the branch (that would place a dealloc
//     after the terminator, which is invalid IR).  Instead the scf.if result
//     is treated as the dealloc root and freed after its last use in the
//     outer scope.
//
// chunk-bytes=1 captures all allocations regardless of size.
//
// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:   -hexagon-dealloc-after-last-use='chunk-bytes=1' | FileCheck %s

// -----

// Buffer pointer extracted, cast through index_cast + inttoptr, then passed
// to an llvm.call.  The call still accesses the memory, so the dealloc must
// come after the call — not after extract_aligned_pointer_as_index.

module attributes {llvm.target_triple = "hexagon"} {
  llvm.func @libfunc(!llvm.ptr, !llvm.ptr, i32) -> i32
      attributes {sym_visibility = "private"}

  func.func @libdevice_ptr_chain(
      %arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c1024 = arith.constant 1024 : i32
    %alloc_in  = memref.alloc() : memref<1024xf32>
    %alloc_out = memref.alloc() : memref<1024xf32>
    memref.copy %arg0, %alloc_in : memref<1024xf32> to memref<1024xf32>
    // Output pointer chain: extract → index_cast → inttoptr → llvm.call arg
    %idx_out = memref.extract_aligned_pointer_as_index %alloc_out
        : memref<1024xf32> -> index
    %i64_out = arith.index_cast %idx_out : index to i64
    %ptr_out  = llvm.inttoptr %i64_out : i64 to !llvm.ptr
    // Input pointer chain: extract → index_cast → inttoptr → llvm.call arg
    %idx_in  = memref.extract_aligned_pointer_as_index %alloc_in
        : memref<1024xf32> -> index
    %i64_in  = arith.index_cast %idx_in : index to i64
    %ptr_in   = llvm.inttoptr %i64_in : i64 to !llvm.ptr
    // The call reads *ptr_in and writes *ptr_out — both buffers must stay live.
    %ret = llvm.call @libfunc(%ptr_in, %ptr_out, %c1024)
        : (!llvm.ptr, !llvm.ptr, i32) -> i32
    memref.copy %alloc_out, %arg1 : memref<1024xf32> to memref<1024xf32>
    return
  }
}

// CHECK-LABEL: func.func @libdevice_ptr_chain
// CHECK:       %[[IN:.+]]  = memref.alloc() : memref<1024xf32>
// CHECK:       %[[OUT:.+]] = memref.alloc() : memref<1024xf32>
// alloc_in's last memref use is the extract, but the pointer is still live
// through index_cast → inttoptr → llvm.call.  No dealloc before the call.
// CHECK:       memref.extract_aligned_pointer_as_index %[[IN]]
// CHECK-NOT:   memref.dealloc
// CHECK:       llvm.call @libfunc
// Dealloc of alloc_in must follow immediately after the call.
// CHECK:       memref.dealloc %[[IN]] : memref<1024xf32>
// alloc_out's last use is the copy below; its dealloc comes after.
// CHECK:       memref.copy %[[OUT]]
// CHECK:       memref.dealloc %[[OUT]] : memref<1024xf32>

// -----

// Buffer allocated in each branch of an scf.if and yielded out.
// No dealloc may appear inside the branches; instead the scf.if result
// is freed after its last use in the outer scope.

func.func @scf_if_yield(
    %cond: i1, %src: memref<1024xf32>, %dst: memref<1024xf32>) {
  %result = scf.if %cond -> (memref<1024xf32>) {
    %buf = memref.alloc() : memref<1024xf32>
    memref.copy %src, %buf : memref<1024xf32> to memref<1024xf32>
    scf.yield %buf : memref<1024xf32>
  } else {
    %buf2 = memref.alloc() : memref<1024xf32>
    scf.yield %buf2 : memref<1024xf32>
  }
  memref.copy %result, %dst : memref<1024xf32> to memref<1024xf32>
  return
}

// CHECK-LABEL: func.func @scf_if_yield
// CHECK:       %[[RES:.+]] = scf.if
// No dealloc anywhere inside the if (neither branch should contain one).
// CHECK-NOT:   memref.dealloc
// CHECK:       }
// Outer scope: copy is the last use of %result; dealloc follows.
// CHECK:       memref.copy %[[RES]]
// CHECK:       memref.dealloc %[[RES]] : memref<1024xf32>

// -----

// Buffer allocated inside an scf.if branch and yielded as a
// reinterpret_cast view.  The pass must follow the reinterpret_cast alias
// to reach the scf.yield, treat the buffer as escaping, and place the
// dealloc outside the if after the last use of the result.

func.func @scf_if_yield_reinterpret_cast(
    %cond: i1, %src: memref<1024xf32>, %dst: memref<512xf32>) {
  %result = scf.if %cond -> (memref<512xf32, strided<[1], offset: ?>>) {
    %buf = memref.alloc() : memref<1024xf32>
    memref.copy %src, %buf : memref<1024xf32> to memref<1024xf32>
    %view = memref.reinterpret_cast %buf to offset: [0], sizes: [512], strides: [1]
        : memref<1024xf32> to memref<512xf32, strided<[1], offset: ?>>
    scf.yield %view : memref<512xf32, strided<[1], offset: ?>>
  } else {
    %buf2 = memref.alloc() : memref<1024xf32>
    %view2 = memref.reinterpret_cast %buf2 to offset: [512], sizes: [512], strides: [1]
        : memref<1024xf32> to memref<512xf32, strided<[1], offset: ?>>
    scf.yield %view2 : memref<512xf32, strided<[1], offset: ?>>
  }
  memref.copy %result, %dst
      : memref<512xf32, strided<[1], offset: ?>> to memref<512xf32>
  return
}

// CHECK-LABEL: func.func @scf_if_yield_reinterpret_cast
// CHECK:       %[[RES:.+]] = scf.if
// No dealloc inside the branches: the reinterpret_cast is an alias of the
// inner alloc; tracing through it reaches scf.yield, so the alloc escapes.
// CHECK-NOT:   memref.dealloc
// CHECK:       }
// The if result (which carries the base pointer of the alloc) is freed
// after its last use in the outer scope.
// CHECK:       memref.copy %[[RES]]
// CHECK:       memref.dealloc %[[RES]]

