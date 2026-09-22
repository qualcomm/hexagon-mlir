//===- Passes.td - Hvx to LLVM passes -----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts hvx dialect ops to llvm.hexagon.* intrinsic calls.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HVXTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_HVXTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def HvxToLLVM : Pass<"hvx-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert the hvx dialect to LLVM intrinsic calls";
  let description = [{
    Lowers `hvx.*` to `llvm.call`s on the `llvm.hexagon.*` intrinsics. Unlike
    the `hmx` leaves, these intrinsics are selected by our own LLVM into HVX
    instructions, so nothing beyond the call is needed: the vector is bitcast
    to the intrinsic's `<32 x i32>` register image, called, and bitcast back
    to the op's lane type.
  }];

  let constructor = "mlir::hvx::createHvxToLLVMPass()";
}

#endif // HEXAGON_CONVERSION_HVXTOLLVM_PASSES_TD
