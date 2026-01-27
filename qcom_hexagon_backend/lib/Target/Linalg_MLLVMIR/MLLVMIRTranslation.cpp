//===-- MLLVMIRTranslation.cpp - Linalg to LLVM IR Translation ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg to LLVM IR translation registration for
// Hexagon target.
//===----------------------------------------------------------------------===//

#include "hexagon/Target/Linalg_MLLVMIR/MLLVMIRTranslation.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/InitAllPasses.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include <dlfcn.h>
#include <filesystem>
#include <iterator>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h"

void setLinalgToLLVMOptions(
    mlir::hexagon::LinalgToLLVMOptions &options,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  // Note: seems very counter-intuitive due to the fact that compare() returns 0
  // when the strings are actually equal, which is why we negate it to convert
  // it to a boolean.
  const std::string TRUE("True");
  options.fusion = !arch_kwargs.at("fusion").compare(TRUE);
  options.fusionAllowRecompute =
      !arch_kwargs.at("fusionAllowRecompute").compare(TRUE);
  options.fusionDoMultiUse = !arch_kwargs.at("fusionDoMultiUse").compare(TRUE);
  options.enableDoubleBuffering =
      !arch_kwargs.at("enableDoubleBuffering").compare(TRUE);
  options.enableSCFThreading =
      !arch_kwargs.at("enableSCFThreading").compare(TRUE);
  options.enableMultiThreading =
      !arch_kwargs.at("enableMultiThreading").compare(TRUE);
  options.enableVTCMTiling = !arch_kwargs.at("enableVTCMTiling").compare(TRUE);
  options.enableConvertToHexagonmem =
      !arch_kwargs.at("enableConvertToHexagonmem").compare(TRUE);
  options.enableHexagonmemCopyToDMA =
      !arch_kwargs.at("enableHexagonmemCopyToDMA").compare(TRUE);
  options.enableHexKL = !arch_kwargs.at("enableHexKL").compare(TRUE);
  options.enableCollapseAddressSpace =
      !arch_kwargs.at("enableCollapseAddressSpace").compare(TRUE);
  options.tileSizes = arch_kwargs.at("tileSizes");
  options.lowerConstantsInSeparateSharedObjects =
      !arch_kwargs.at("lowerConstantsInSeparateSharedObjects").compare(TRUE);
  options.enableBufferization =
      !arch_kwargs.at("enableBufferization").compare(TRUE);
  options.enableSplitReduction =
      !arch_kwargs.at("enableSplitReduction").compare(TRUE);
  options.enableConvTiling = !arch_kwargs.at("enableConvTiling").compare(TRUE);
  options.convTilingFactor = std::stoi(arch_kwargs.at("convTilingFactor"));
  options.convTileHeightDim =
      !arch_kwargs.at("convTileHeightDim").compare(TRUE);
  options.convTileOCDim = !arch_kwargs.at("convTileOCDim").compare(TRUE);
  options.enableLWP = !arch_kwargs.at("enableLWP").compare(TRUE);
  options.disableLWPLoop = !arch_kwargs.at("disableLWPLoop").compare(TRUE);
  options.enableVectorization =
      !arch_kwargs.at("enableVectorization").compare(TRUE);
}

namespace mlir {
namespace hexagon {

mlir::ModuleOp translateLinalgToLLVMMLIR(
    mlir::ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {
  mlir::PassManager pm(mod->getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return nullptr;
  }

  auto printingFlags = mlir::OpPrintingFlags();
  if (mlir::hexagon::isEnvTrue("MLIR_ELIDE_LARGE_CONST_PRINT")) {
    printingFlags.elideLargeElementsAttrs(1);
    printingFlags.elideLargeResourceString(1);
  } else {
    printingFlags.elideLargeElementsAttrs(16);
  }
  // Print the IR after HexagonLWPPass if enabled for debug purpose
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return mlir::hexagon::isEnvTrue("MLIR_ENABLE_DUMP") ||
               llvm::StringRef(pass->getName()).contains("HexagonLWP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  // set your enable/disable individual pass options here
  // or funnel to here.
  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);
  pm.addPass(createLinalgToLLVMPass(options));

  if (failed(pm.run(mod))) {
    llvm::errs() << "Linalg to Hexagon Pass execution failed";
    return nullptr;
  }
  return mod;
}

// -------------------------------------------------
// --- Translating to multiple LLVM/MLIR modules ---
// -------------------------------------------------

class CustomPassManager : public mlir::PassManager {
public:
  CustomPassManager(mlir::MLIRContext *context, LinalgToLLVMOptions options)
      : mlir::PassManager(context) {
    addPass(createLinalgToLLVMPass(options));

    // Careful: here we are giving ownership on this pass, so we can't access it
    // ourselves anymore directly
    addPass(std::make_unique<LowerConstantsSeparatelyPass>());
  }

  std::vector<ModuleOp> getProducedModules() const {
    LowerConstantsSeparatelyPass *ptr_lowerConstantsPass;
    auto passes = getPasses();

    // Trick to get the LowerConstantsSeparatelyPass since we had given
    // ownership to it. It's just some plain bureaucracy: finding the pass
    // LowerConstantsSeparatelyPass amongst all the passes that were added to
    // the pass manager
    for (auto &pass : passes) {
      if ((ptr_lowerConstantsPass =
               dynamic_cast<LowerConstantsSeparatelyPass *>(&pass))) {
        // We found the LowerConstantsSeparatelyPass pass in the pass manager
        break;
      }
    }
    if (!ptr_lowerConstantsPass) {
      std::cerr << "Error: pass_lower_constants_separately is null!"
                << std::endl;
      return {};
    }

    // Returning the list of modules the pass LowerConstantsSeparatelyPass has
    // produced
    return ptr_lowerConstantsPass->getProducedModules();
  }
};

std::vector<ModuleOp> translateLinalgToMultipleLLVMMLIRModules(
    ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);

  CustomPassManager pm(mod->getContext(), options);
  if (failed(pm.run(mod))) {
    llvm::errs() << "Custom pass manager for producing multiple modules failed";
    return std::vector<ModuleOp>();
  }

  std::vector<ModuleOp> all_modules = pm.getProducedModules();
  // Insert the main module (mutated) at the front of the vector that will be
  // returned
  all_modules.insert(all_modules.begin(), mod);
  return all_modules;
}

} // namespace hexagon
} // namespace mlir
