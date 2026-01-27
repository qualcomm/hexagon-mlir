//===- Transforms.h - Linalg lowering and optimization passes -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_TRANSFORMS_H
#define HEXAGON_TRANSFORMS_TRANSFORMS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
#define GEN_PASS_DECL
#include "hexagon/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createCollapseAddressSpacePass();

std::unique_ptr<OperationPass<func::FuncOp>> createCopyCanonicalizationPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createConvTilingPass(const ConvTilingOptions &options = ConvTilingOptions());
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLayoutPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertToHexagonmemPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertZeroSizeMemrefPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeTensorConcatPass();

std::unique_ptr<OperationPass<ModuleOp>> createEraseUnusedLinalgOperands();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createExpandBoolVecPass();

std::unique_ptr<OperationPass<func::FuncOp>> createExpandMathOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createFastInversePass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonAddFastMathPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS1Pass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS2Pass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonLLVMEnableHexagonRoutinesPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonLowerTmTensorPass();

std::unique_ptr<OperationPass<func::FuncOp>> createHexagonLWPPass(
    const HexagonLWPPassOptions &options = HexagonLWPPassOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonPuntBufferPass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonReplaceWithLibraryCallsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexagonRVOPass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonVectorLoweringPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgGeneralizePass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerLibdevicePass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTPtrPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTTXPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMatmulToConvPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMatmulToHexKLPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createDecomposeHexKLMatmulPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMemoryOffsetsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createScheduleMatmulForHVXPass();

std::unique_ptr<OperationPass<ModuleOp>> createSmallExponentToMultiplyPass(
    const SmallExponentToMultiplyOptions &options =
        SmallExponentToMultiplyOptions());

std::unique_ptr<OperationPass<ModuleOp>> removeMLProgramPass();

std::unique_ptr<Pass> createReduceContractionRankPass();

std::unique_ptr<Pass> createHoistScalarOpsPass();

std::unique_ptr<Pass> createFoldMulFByZeroPass();

} // namespace hexagon
} // namespace mlir

#endif //  HEXAGON_TRANSFORMS_TRANSFORMS_H
