//===- Passes.td - Hmx to LLVM passes ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts hmx dialect ops to the address-only runtime leaves.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HMXTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_HMXTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def HmxToLLVM : Pass<"hmx-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert the hmx dialect to LLVM calls";
  let description = [{
    Lowers `hmx.*` to `llvm.call`s on the runtime leaves, which take addresses
    rather than pointers (the runtime is compiled with the SDK's clang, so its
    ABI is plain integers; see docs/hmx/hmx-system-design.md §3 decision #2, prebuilt static library).

    A crouton operand becomes `ptrtoint` of the memref's aligned pointer. The
    function names live in HmxExternalFnNames, one table for the whole dialect.

    `hmx.acc_read` lowers to *two* calls -- the bias registers have to be loaded
    for the selected set before the read-out -- because that pairing is the op's
    contract rather than the caller's business.
  }];

  let constructor = "mlir::hmx::createHmxToLLVMPass()";
}

#endif // HEXAGON_CONVERSION_HMXTOLLVM_PASSES_TD
