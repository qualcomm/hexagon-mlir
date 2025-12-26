//===-- LLVMIRTranslation.cpp - Hexagon LLVM IR Translation ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon target LLVM IR translation registration.
//===----------------------------------------------------------------------===//

#include "hexagon/Target/HEX_LLVMIR/LLVMIRTranslation.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

namespace mlir {
namespace hexagon {

static bool linkHexExternLib(llvm::Module &module, llvm::StringRef name,
                             llvm::StringRef path) {
  return true;
  // nothing for now.
}

void initializeHexagonTarget() {
  static bool isInitialized = false;
  if (isInitialized)
    return;
  isInitialized = true;
  // Initialize LLVM targets for Hexagon.
  LLVMInitializeHexagonTargetInfo();
  LLVMInitializeHexagonTarget();
  LLVMInitializeHexagonTargetMC();
  LLVMInitializeHexagonAsmParser();
  LLVMInitializeHexagonAsmPrinter();
}

void dumpHexagonAssembly(llvm::Module &module, unsigned moduleId,
                         const std::string &archVersion) {
  initializeHexagonTarget();
  // Construct output file name.
  // Check if env var HEXAGON_ASM_DUMP_FILE is set to override default name.
  const char *envAsmFile = std::getenv("HEXAGON_ASM_DUMP_FILE");
  std::string fileName;
  if (envAsmFile) {
    fileName = std::string(envAsmFile);
  } else {
    fileName = "hexagon_module_" + std::to_string(moduleId);
  }
  std::string asmFileName = fileName + ".s";
  llvm::errs() << "Dumping Hexagon assembly to " << asmFileName << "\n";
  // Remove the assembly file if it already exists.
  llvm::sys::fs::remove(asmFileName);

  // Determine target triple and architecture.
  std::string archVer = archVersion;
  if (archVer.front() != 'v')
    archVer = "v" + archVer;

  std::string cpu = "hexagon" + archVer;
  std::string features =
      "+hvx" + archVer + ",+hvx-length128b"; // Modify as needed.
  auto targetTriple = module.getTargetTriple();

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Target lookup failed: " << error << "\n";
    return;
  }

  llvm::TargetOptions opt;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, cpu, features, opt, {}));

  auto emitFile = [&](llvm::CodeGenFileType fileType,
                      const std::string &filename, const char *label) {
    std::error_code ec;
    llvm::raw_fd_ostream out(filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Error opening " << label
                   << " file for writing: " << ec.message() << "\n";
      return;
    }
    llvm::legacy::PassManager pass;
    if (targetMachine->addPassesToEmitFile(pass, out, nullptr, fileType)) {
      llvm::errs() << "Target doesn't support generation of this file type\n";
      return;
    }
    pass.run(module);
    out.flush();
    llvm::errs() << "[Info] " << label << " emitted to " << filename << "\n";
  };

  // Emit assembly to file.
  emitFile(llvm::CodeGenFileType::AssemblyFile, asmFileName, "Assembly");

  // Emit object file if HEXAGON_ASM_TO_OBJ is set.
  if (std::getenv("HEXAGON_ASM_TO_OBJ")) {
    std::string objFileName = fileName + ".o";
    llvm::errs() << "Emitting object file to " << objFileName << "\n";
    std::vector<char> objBytes = llvm_module_to_obj_string(module);

    std::error_code ec;
    llvm::raw_fd_ostream objOut(objFileName, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Error opening object file for writing: " << ec.message()
                   << "\n";
      return;
    }
    objOut.write(objBytes.data(), objBytes.size());
    objOut.flush();

    llvm::errs() << "[Info] Object file emitted to " << objFileName << "\n";
  }
}

static std::unique_ptr<llvm::Module> translateLLVMToLLVMIR(
    llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
    const std::unordered_map<std::string, std::string> &arch_kwargs,
    LLVMOptimizationLevel optLevel) {
  DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }
  llvmModule->setTargetTriple(llvm::Triple(arch_kwargs.at("arch_triple")));
  llvmModule->setDataLayout(arch_kwargs.at("data_layout"));
  for (llvm::Function &f : llvmModule->functions())
    f.addFnAttr("target-features", arch_kwargs.at("arch_features"));

  if (optLevel != LLVMOptimizationLevel::NoOptimization) {
    auto optPipeline = mlir::makeOptimizingTransformer(
        static_cast<int>(optLevel), /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
      return nullptr;
    }
  }

  return llvmModule;
}

std::unique_ptr<llvm::Module> translateHexagonMlirLlvmToLLVMIR(
    llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
    const std::unordered_map<std::string, std::string> &arch_kwargs,
    LLVMOptimizationLevel optLevel) {

  auto llvmIR =
      translateLLVMToLLVMIR(llvmContext, module, arch_kwargs, optLevel);
  if (!llvmIR) {
    llvm::errs() << "Translate to LLVM IR failed";
    return nullptr;
  }

  return llvmIR;
}

static std::unique_ptr<llvm::TargetMachine>
create_target_machine(const llvm::Triple &target_triple) {

  // Hexagon target instance is needed to give a broader description of the
  // target platform
  std::string error;
  const llvm::Target *target_instance =
      llvm::TargetRegistry::lookupTarget(target_triple, error);
  if (!target_instance) {
    llvm::report_fatal_error(
        "Target_triple" + llvm::Twine(target_triple.str()) +
        "is not supported failed to create TargetInstance");
  }

  // Create the target machine using the target.
  // Provides a detailed configuration for generating optimized machine code.
  const llvm::TargetOptions &target_options = {};
  const llvm::Reloc::Model &reloc_model = llvm::Reloc::PIC_;
  const llvm::CodeModel::Model &code_model = llvm::CodeModel::Small;
  const llvm::CodeGenOptLevel &codegen_optlevel =
      llvm::CodeGenOptLevel::Aggressive;

  llvm::TargetMachine *targetMachine = target_instance->createTargetMachine(
      target_triple, "", "", target_options, reloc_model, code_model,
      codegen_optlevel);

  if (!targetMachine)
    llvm::report_fatal_error("Error while creating create TargetMachine");
  return std::unique_ptr<llvm::TargetMachine>(targetMachine);
}

inline std::vector<char> llvm_module_to_obj_string(llvm::Module &llvmModule) {
  auto dummy = std::unique_ptr<llvm::Module>(&llvmModule);
  auto result = llvm_module_to_obj_string(dummy);
  dummy.release();
  return result;
}

/// Generate object file from mlir-llvm module.
std::vector<char>
llvm_module_to_obj_string(std::unique_ptr<llvm::Module> &llvmModule) {

  llvm::SmallVector<char, 0> binaryData; // Will grow dyamically as needed.
  // Create a raw_svector_ostream that uses the SmallVector as
  // its buffer for binary data.
  llvm::raw_svector_ostream stream(binaryData);

  initializeHexagonTarget();

  auto target_triple = llvmModule->getTargetTriple();
  std::unique_ptr<llvm::TargetMachine> targetMachine =
      create_target_machine(target_triple);
  llvm::legacy::PassManager codegenPasses;
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // 'addPassesToEmitFile' adds the necessary passes to the pass manager
  // to transform the LLVM IR into the specified file type.
  // If successful, the output will be written to the raw_svector_ostream.
  if (targetMachine->addPassesToEmitFile(codegenPasses, stream, nullptr,
                                         llvm::CodeGenFileType::ObjectFile))
    llvm::errs() << "Error while emitting target code. \n";

  codegenPasses.run(*llvmModule);

  // Once the pass runs and the value is populated in the SmallVector,
  // the data is dumped to obj_bc (vector<char>).
  // This format is helpful for binary data.
  std::vector<char> obj_bc;
  obj_bc = std::vector<char>(binaryData.begin(), binaryData.end());
  if (obj_bc.empty())
    llvm::errs() << "The object string dumped for the llvm module for this "
                    "target is empty. \n";

  // We prefer returning vector of char since byte data is saved and
  // vector<char> is similar to storing array of bytes.
  return obj_bc;
}

} // namespace hexagon
} // namespace mlir
