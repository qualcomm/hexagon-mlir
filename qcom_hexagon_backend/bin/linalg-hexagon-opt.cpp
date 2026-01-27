//===- linalg-hexagon-opt.cpp ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_ostream.h"

#include "hexagon/Conversion/AffineToLLVM/Passes.h"
#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexKL/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);   // TODO: restrict
  mlir::registerAllPasses();             // TODO: restrict
  mlir::registerAllExtensions(registry); // TODO: restrict

  registry.insert<mlir::ttx::TTXDialect>();
  registry.insert<mlir::tm_tensor::TmTensorDialect>();
  registry.insert<mlir::tptr::HexagonTPtrDialect>();
  registry.insert<mlir::hexagonmem::HexagonMemDialect>();
  registry.insert<mlir::hexkl::HexKLDialect>();

  mlir::hexagonmem::registerConvertHexagonMemToLLVMInterface(registry);

  mlir::hexagon::registerAffineToLLVMPass();
  mlir::hexagon::registerAffineTilingPass();
  mlir::hexagon::registerAffinePipelineFusionPass();
  mlir::hexagon::registerAffineVectorizePass();
  mlir::hexagon::registerAffineTileMemoryPass();

  mlir::hexagon::registerHexagonAddFastMathPass();
  mlir::hexagon::registerHexagonFusionPass();
  mlir::hexagon::registerEraseUnusedLinalgOperands();
  mlir::hexagon::registerReduceContractionRankPass();
  mlir::hexagon::registerHoistScalarOpsPass();
  mlir::hexagon::registerScheduleMatmulForHVXPass();
  mlir::hexagon::registerLinalgGeneralizePass();
  mlir::hexagon::registerHexagonDoubleBufferGenericS1Pass();
  mlir::hexagon::registerHexagonDoubleBufferGenericS2Pass();
  mlir::hexagon::registerHexagonSlicingPass();
  mlir::hexagon::registerHexagonTilingPass();
  mlir::hexagon::registerLinalgToLLVMPass();
  mlir::hexagon::registerHexagonVectorLoweringPass();
  mlir::hexagon::registerHexagonVectorizationPass();
  mlir::hexagon::registerRewriteUBPoisonToZeroPass();
  mlir::hexagon::registerHexagonReplaceWithLibraryCallsPass();
  mlir::hexagon::registerHexagonPuntBufferPass();
  mlir::hexagon::registerHexagonRVOPass();
  mlir::hexagon::registerDecomposeTensorConcatPass();
  mlir::hexagon::registerMatmulToConvPass();
  mlir::hexagon::registerMatmulToHexKLPass();
  mlir::hexagon::registerDecomposeHexKLMatmulPass();
  mlir::hexagon::registerConvTilingPass();
  mlir::hexagon::registerDMAToLLVMPass();
  mlir::hexagon::registerExpandMathOpsPass();
  mlir::hexagon::registerRemoveMLProgramPass();
  mlir::hexagon::registerHexagonLLVMEnableHexagonRoutines();
  mlir::hexagonmem::registerHexagonMemToLLVMPass();
  mlir::hexkl::registerHexKLToLLVMPass();
  mlir::hexagon::registerExpandBoolVecPass();
  mlir::hexagon::registerLowerConstantsSeparatelyPass();
  mlir::hexagon::registerHexmemCpyToDMA();
  mlir::hexagon::registerVTCMTilingPass();
  mlir::hexagon::registerConvertToHexagonmemPass();
  mlir::hexagon::registerCollapseAddressSpacePass();
  mlir::hexagon::registerConvertLayoutPass();
  mlir::hexagon::registerFastInversePass();
  mlir::hexagon::registerConvertZeroSizeMemref();
  mlir::hexagon::registerSmallExponentToMultiplyPass();
  mlir::hexagon::registerFormSCFThreadsPass();
  mlir::hexagon::registerFormVirtualThreadsPass();
  mlir::hexagon::registerFormAsyncThreadsPass();
  mlir::hexagon::registerMemoryOffsetsPass();
  mlir::hexagon::registerCopyCanonicalizationPass();
  mlir::hexagon::registerHexagonLWPPass();
  mlir::hexagon::registerHexagonLowerTmTensorPass();
  mlir::hexagon::registerLowerTTXPass();
  mlir::hexagon::registerLowerTPtrPass();
  mlir::hexagon::registerFoldMulFByZeroPass();

  // Register all external models.
  mlir::hexkl::registerBufferizableOpInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Linalg To Hexagon LLVM compiler driver\n", registry));
}
