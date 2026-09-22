//===- HvxToLLVM.h - Lower hvx ops to LLVM ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_CONVERSION_HVXTOLLVM_HVXTOLLVM_H
#define HEXAGON_CONVERSION_HVXTOLLVM_HVXTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hvx {

#define GEN_PASS_DECL
#include "hexagon/Conversion/HvxToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createHvxToLLVMPass();

} // namespace hvx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HVXTOLLVM_HVXTOLLVM_H
