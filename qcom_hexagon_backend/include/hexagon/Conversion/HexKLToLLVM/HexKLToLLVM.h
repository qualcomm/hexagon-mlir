//===- HexKLToLLVM.h   - Lower hexkl ops to LLVM --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLTOLLVM_H
#define HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexkl {

#define GEN_PASS_DECL
#include "hexagon/Conversion/HexKLToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createHexKLToLLVMPass();

void registerConvertHexKLToLLVMInterface(DialectRegistry &registry);

} // namespace hexkl
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLTOLLVM_H
