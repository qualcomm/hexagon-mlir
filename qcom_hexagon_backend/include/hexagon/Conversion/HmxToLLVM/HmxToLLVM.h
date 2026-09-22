//===- HmxToLLVM.h - Lower hmx ops to LLVM --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_CONVERSION_HMXTOLLVM_HMXTOLLVM_H
#define HEXAGON_CONVERSION_HMXTOLLVM_HMXTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hmx {

#define GEN_PASS_DECL
#include "hexagon/Conversion/HmxToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createHmxToLLVMPass();

void populateHmxToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HMXTOLLVM_HMXTOLLVM_H
