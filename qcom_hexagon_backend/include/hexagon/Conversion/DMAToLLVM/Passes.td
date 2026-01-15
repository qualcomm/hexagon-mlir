//===- Passes.td - DMA to LLVM passes ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts memref.dma to llvm ops.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_DMATOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_DMATOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def DMAToLLVM :
    Pass<"dma-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert memref.dma ops to LLVM";

  let constructor = "mlir::hexagon::createDMAToLLVMPass()";

  let dependentDialects = ["memref::MemRefDialect", "mlir::LLVM::LLVMDialect"];

}

#endif // HEXAGON_CONVERSION_DMATOLLVM_PASSES_TD
