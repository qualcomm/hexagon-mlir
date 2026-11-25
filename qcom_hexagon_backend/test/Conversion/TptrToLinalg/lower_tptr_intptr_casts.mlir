//===- lower_tptr_intptr_casts.mlir ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-tptr))' | FileCheck %s

// tptr.inttoptr / tptr.ptrtoint are emitted by triton_shared when lowering
// tt.int_to_ptr / tt.ptr_to_int on the scalar pointer path (e.g. loading a raw
// pointer stored as an integer out of a pointer array). lower-tptr rewrites them
// to the same index<->pointer materialization used by the other tptr ops.

// inttoptr: integer -> !ptr.ptr, lowered to an index_cast plus an
// unrealized_conversion_cast to the pointer type. The result feeds ptr.from_ptr
// (handled by lower-core-ptr).
// CHECK-LABEL:   func.func @inttoptr
// CHECK-SAME:      (%[[I:.*]]: i64)
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[I]] : i64 to index
// CHECK:           %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[MR:.*]] = ptr.from_ptr %[[PTR]]
func.func @inttoptr(%i : i64) -> memref<1xf32> {
  %p  = tptr.inttoptr %i : i64 to <#tptr.default_memory_space>
  %m  = ptr.from_ptr %p : <#tptr.default_memory_space> -> memref<1xf32, #tptr.default_memory_space>
  %ms = memref.memory_space_cast %m : memref<1xf32, #tptr.default_memory_space> to memref<1xf32>
  return %ms : memref<1xf32>
}

// ptrtoint: !ptr.ptr -> integer, lowered to the pointer's index materialization
// (here via from_memref) followed by an index_cast to the requested width.
// CHECK-LABEL:   func.func @ptrtoint
// CHECK:           %[[IDX:.*]] = memref.extract_aligned_pointer_as_index
// CHECK:           %[[INT:.*]] = arith.index_cast %[[IDX]] : index to i64
// CHECK:           return %[[INT]] : i64
func.func @ptrtoint(%base : memref<1xi32>) -> i64 {
  %p = tptr.from_memref %base : memref<1xi32> to <#tptr.default_memory_space>
  %i = tptr.ptrtoint %p : <#tptr.default_memory_space> to i64
  return %i : i64
}
