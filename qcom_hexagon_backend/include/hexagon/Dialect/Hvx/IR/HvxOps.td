//===- HvxOps.td - HVX dialect ops --------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HVX_IR_HVX_OPS_TD
#define HEXAGON_DIALECT_HVX_IR_HVX_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "hexagon/Dialect/Hvx/IR/HvxDialect.td"

class Hvx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hvx_Dialect, mnemonic, traits> {
}

// A 1-D vector occupying exactly one 128-byte HVX vector register. The ODS
// constraint only pins the kind; the exact "1-D and 128 bytes" contract is the
// op verifier's (which can spell out the byte count in its message).
def Hvx_IsVector : CPred<"::llvm::isa<::mlir::VectorType>($_self)">;
def Hvx_HvxVector : Type<Hvx_IsVector, "128-byte 1-D vector">;

def Hvx_VrorOp : Hvx_Op<"vror", [Pure, SameOperandsAndResultType]> {
  let summary = "Rotate one 128-byte HVX vector right by a whole number of bytes";

  let description = [{
    Byte-exact image of the HVX `Vd = vror(Vu, Rt)` instruction (V79 HVX
    Programmer Reference Manual, permute section):

    ```
    for (k = 0; k < VWIDTH; k++)
      Vd.ub[k] = Vu.ub[(k + Rt) & (VWIDTH - 1)];
    ```

    i.e. a circular right rotation of the whole 128-byte register, where the
    lane that was `bytes` positions to the left wraps around to the front. It
    is element-type agnostic -- the instruction permutes bytes -- so the op
    carries any 1-D vector type whose total byte width is 128.

    The rotate amount is a compile-time byte count (the instruction itself
    takes a general register, so no value is unencodable; the immediate keeps
    the op cheap and lets the butterfly users stay readable). Downstream it
    lowers to `llvm.hexagon.V6.vror.128B`, which the Hexagon backend selects
    into `V6_vror`.

    Motivating use: the last stage of a row reduction. Rotating the
    accumulator by half the register and folding (max/add) halves the number
    of lanes holding the running reduction value; after log2(128/elemBytes)
    steps every lane holds the row's reduction, which is exactly the
    llama.cpp `hvx-reduce.h` butterfly.
  }];

  let arguments = (ins Hvx_HvxVector:$input, I32Attr:$bytes);
  let results = (outs Hvx_HvxVector:$result);

  let hasVerifier = 1;

  let assemblyFormat = "$input `,` $bytes attr-dict `:` type($input)";
}

#endif // HEXAGON_DIALECT_HVX_IR_HVX_OPS_TD
