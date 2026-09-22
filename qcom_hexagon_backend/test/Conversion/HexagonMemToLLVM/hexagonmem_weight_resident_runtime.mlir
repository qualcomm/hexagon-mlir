//===- hexagonmem_weight_resident_runtime.mlir - runtime resident lowering -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A runtime (non-constant) resident weight carries its source address as the
// alloc's extra operand (the residency key the runtime pins on), and lowers to
// the same runtime entry the constant path uses.
//
// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @hexagon_runtime_weight_resident_dsp(i64, i32) -> !llvm.ptr

module {
  // CHECK-LABEL: func.func @resident_runtime
  func.func @resident_runtime() {
    %addr = arith.constant 4096 : index
    // The residency key operand (the weight's aligned pointer) becomes the
    // call's first argument. The index operand is carried into i64 by the
    // surrounding conversion.
    // CHECK: llvm.call @hexagon_runtime_weight_resident_dsp({{.*}}, {{.*}}) : (i64, i32) -> !llvm.ptr
    %w = hexagonmem.alloc(%addr) {alignment = 128 : i64, hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
    return
  }
}
