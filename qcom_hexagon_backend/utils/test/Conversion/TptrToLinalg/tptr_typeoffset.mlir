// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-tptr),convert-to-llvm,cse,canonicalize)' | FileCheck %s

module {
func.func @type_offseti32_i32() -> i32 {
// CHECK-LABEL:   llvm.func @type_offseti32_i32() -> i32 {
  %r = tptr.type_offset i32  : i32
// CHECK-NEXT:    {{.+}} = llvm.mlir.constant(4 : i32) : i32
  return %r : i32
}

func.func @type_offseti32_i64() -> i64 {
  // CHECK-LABEL:   llvm.func @type_offseti32_i64() -> i64 {
  %r = tptr.type_offset i32  : i64
  // CHECK-NEXT:    {{.+}} = llvm.mlir.constant(4 : i64) : i64
  return %r : i64
}

func.func @type_offsetf16_i64() -> i64 {
  // CHECK-LABEL:   llvm.func @type_offsetf16_i64() -> i64 {
  %r = tptr.type_offset f16  : i64
  // CHECK-NEXT:    {{.+}} = llvm.mlir.constant(2 : i64) : i64
  return %r : i64
}

func.func @type_offsetf16_index() -> index {
// CHECK-LABEL:   llvm.func @type_offsetf16_index() -> i64 {
  %r = tptr.type_offset f16  : index
// CHECK-NEXT:    {{.+}} = llvm.mlir.constant(2 : index) : i64
  return %r : index
}
}
