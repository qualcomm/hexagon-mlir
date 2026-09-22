//===- HvxDialect.td - HVX dialect --------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The `hvx` dialect: thin wrappers over single HVX instructions that have no
// generic MLIR/LLVM equivalent. Ops lower to `llvm.call`s on `llvm.hexagon.*`
// intrinsics, which our own LLVM selects into the instructions -- no runtime
// library involved (that is the difference from `hmx`, whose engine builtins
// only the SDK's clang can lower).
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_TD
#define HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_TD

include "mlir/IR/OpBase.td"

def Hvx_Dialect : Dialect {
  let name = "hvx";

  let cppNamespace = "::mlir::hvx";

  let summary = "HVX vector primitives";

  let description = [{
    The `hvx` dialect exposes single Hexagon HVX instructions that the vector
    pipeline cannot express with generic ops, so a lowering pass can spell them
    out instead of letting the scalarized fallback run.

    Ops are element-type agnostic where the underlying instruction is a byte
    permutation: the instruction operates on one 128-byte vector register
    regardless of the lane type riding on it.
  }];
}

#endif // HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_TD
