//===- HexagonMemToLLVM.h - Lower hexagonmem ops to LLVM ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMTOLLVM_H
#define HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagonmem {

#define GEN_PASS_DECL
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createHexagonMemToLLVMPass();

void registerConvertHexagonMemToLLVMInterface(DialectRegistry &registry);

} // namespace hexagonmem
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMTOLLVM_H
