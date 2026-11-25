//===- core_ptr_typeoffset.mlir -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(finalize-memref-and-core-ptr-to-llvm-conversion{enable-collapse-address-space=false})' | FileCheck %s

// ptr.type_offset lowers to an arith.constant equal to sizeof(elementType).
// Mirrors the tptr.type_offset test in TptrToLinalg/tptr_typeoffset.mlir.

module {
func.func @type_offseti32_i32() -> i32 {
// CHECK-LABEL:   func.func @type_offseti32_i32() -> i32 {
  %r = ptr.type_offset i32 : i32
// CHECK-NEXT:    {{.+}} = arith.constant 4 : i32
  return %r : i32
}

func.func @type_offseti32_i64() -> i64 {
// CHECK-LABEL:   func.func @type_offseti32_i64() -> i64 {
  %r = ptr.type_offset i32 : i64
// CHECK-NEXT:    {{.+}} = arith.constant 4 : i64
  return %r : i64
}

func.func @type_offsetf16_i64() -> i64 {
// CHECK-LABEL:   func.func @type_offsetf16_i64() -> i64 {
  %r = ptr.type_offset f16 : i64
// CHECK-NEXT:    {{.+}} = arith.constant 2 : i64
  return %r : i64
}

func.func @type_offsetf16_index() -> index {
// CHECK-LABEL:   func.func @type_offsetf16_index
// NOTE: convert-func-to-llvm converts the function boundary (index -> i64)
// but does not touch ptr.type_offset itself, so a leftover
// unrealized_conversion_cast is expected here; verify against actual output.
// CHECK:         {{.+}} = arith.constant 2 : index
  %r = ptr.type_offset f16 : index
  return %r : index
}
}
