//===- flatten-zero-loop-assert.mlir --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -flatten-zero-loop-linalg | FileCheck %s

// Triton can emit a side-effect-only assertion wrapper as a zero-loop
// linalg.generic.  In particular, the wrapper has no DPS operands, results,
// or region arguments, while its body contains control flow.  It must be
// flattened before SCF/control-flow lowering so the linalg region does not
// become multi-block.
module {
  func.func private @assert_side_effect()

  func.func @zero_loop_assert(%arg0: i1) {
    "linalg.generic"() ({
    ^bb0:
      scf.if %arg0 {
        cf.assert %arg0, "assertion"
        func.call @assert_side_effect() : () -> ()
      }
      linalg.yield
    }) {indexing_maps = [], iterator_types = []} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @zero_loop_assert
// CHECK-NOT: linalg.generic
// CHECK: scf.if
// CHECK: cf.assert
// CHECK: func.call @assert_side_effect
