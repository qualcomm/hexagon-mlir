//===- AffineToLLVM.h -----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_AFFINETOLLVM_AFFINETOLLVM_H
#define HEXAGON_CONVERSION_AFFINETOLLVM_AFFINETOLLVM_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
#define GEN_PASS_DECL
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

// affine-to-llvm aggregate pass.
std::unique_ptr<OperationPass<ModuleOp>> createAffineToLLVMPass(
    const AffineToLLVMOptions &options = AffineToLLVMOptions());

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAffineTilingPass(
    const AffineTilingOptions &options = AffineTilingOptions());

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createAffinePipelineFusionPass();

// affine vectorize pass.
std::unique_ptr<OperationPass<func::FuncOp>> createAffineVectorizePass();

std::unique_ptr<OperationPass<func::FuncOp>> createAffineTileMemoryPass(
    const AffineTileMemoryOptions &options = AffineTileMemoryOptions());

} // namespace hexagon
} // namespace mlir

#endif //  HEXAGON_CONVERSION_AFFINETOLLVM_AFFINETOLLVM_H
