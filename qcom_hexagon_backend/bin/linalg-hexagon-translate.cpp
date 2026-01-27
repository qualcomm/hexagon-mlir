//===- linalg-hexagon-translate.cpp - Tool for opt and translate ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This tool enables one to translate from hexagon mlir-llvm to llvm-ir.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "hexagon/Target/HEX_LLVMIR/LLVMIRTranslation.h"

#include "LinkRuntimeModules.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdlib>
#include <iostream>

namespace mlir::hexagon {
namespace { // unnamed

OwningOpRef<ModuleOp> loadMLIRModule(llvm::StringRef inputFilename,
                                     MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  mlir::DialectRegistry registry;
  registry.insert<math::MathDialect, arith::ArithDialect,
                  mlir::LLVM::LLVMDialect, scf::SCFDialect>();

  context.appendDialectRegistry(registry);

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer)
      -> OwningOpRef<ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects();

    OwningOpRef<ModuleOp> module =
        parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Parse MLIR file failed.";
      return nullptr;
    }

    return module;
  };

  auto module = processBuffer(std::move(input));
  if (!module)
    return nullptr;

  return module;
}

enum class EmitKind { LLVMIR, Obj };

LogicalResult translateMain(int argc, char **argv, llvm::StringRef toolName) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  // Note that if the option "-o" isn't provided, its default value is "-"
  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<EmitKind> emitKind(
      "emit", llvm::cl::desc("Translation target"),
      llvm::cl::values(clEnumValN(EmitKind::LLVMIR, "llvmir", "Emit LLVM IR"),
                       clEnumValN(EmitKind::Obj, "obj", "Emit object file")),
      llvm::cl::init(EmitKind::LLVMIR));

  static llvm::cl::opt<std::string> targetTriple(
      "triple", llvm::cl::desc("target triple. e.g. '-hexagon-unknown-elf'"),
      llvm::cl::value_desc("target triple"), llvm::cl::init("hexagon"));

  const char *hexArchVersion = std::getenv("HEXAGON_ARCH_VERSION");
  if (!hexArchVersion) {
    llvm::errs() << "Please set HEXAGON_ARCH_VERSION env\n";
    return failure();
  }
  std::string featureStr =
      "+hvxv" + std::string(hexArchVersion) + ",+hvx-length128b";
  static llvm::cl::opt<std::string> targetFeatures(
      "features",
      llvm::cl::desc("target features. e.g. '+hvxv68,+hvx-length128b'"),
      llvm::cl::value_desc("target features"), llvm::cl::init(featureStr));
  static llvm::cl::opt<std::string> targetDL(
      "dl", llvm::cl::desc("target data layout"),
      llvm::cl::value_desc("target DL"),
      llvm::cl::init("e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:"
                     "16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:"
                     "512:512-v1024:1024:1024-v2048:2048:2048"));

  static llvm::cl::opt<bool> noHexagonRuntime(
      "no-hexagon-runtime",
      llvm::cl::desc("Disable implicit linking of Hexagon Runtime during"
                     "translation"),
      llvm::cl::init(false));

  llvm::InitLLVM y(argc, argv);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  std::unordered_map<std::string, std::string> arch_kwargs = {
      {"arch_triple", targetTriple},
      {"arch_features", targetFeatures},
      {"data_layout", targetDL}};

  mlir::MLIRContext context;
  auto module = loadMLIRModule(inputFilename, context);
  if (!module)
    return failure();

  std::string errorMessage;
  // Open outputFilename if it was provided, or the standard output if it's "-"
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto llvmIR = translateHexagonMlirLlvmToLLVMIR(
      &llvmContext, *module, arch_kwargs, LLVMOptimizationLevel::O3);
  if (!llvmIR) {
    llvm::errs() << "Translate to LLVM IR failed\n";
    return failure();
  }

  if (!noHexagonRuntime)
    mlir::Hexagon::Translate::linkRuntimeModules(llvmContext, llvmIR);

  if (emitKind == EmitKind::Obj) {
    std::vector<char> object_code_as_bytes = llvm_module_to_obj_string(llvmIR);
    // Dump object_code_as_bytes to output here
    output->os().write(object_code_as_bytes.data(),
                       object_code_as_bytes.size());
  } else {
    // Write to the 'output' (which is either a file if provided by the -o
    // option, or the standard output if non-provided)
    llvmIR->print(output->os(), nullptr);
  }
  // Ensure all the data is written out
  output->os().flush();
  // Indicate that the output file should not be deleted (LLVM has a notion of
  // temp file vs files to keep)
  output->keep();

  return success();
}

} // namespace
} // namespace mlir::hexagon

int main(int argc, char **argv) {
  return failed(mlir::hexagon::translateMain(
      argc, argv, "Qualcomm mlir-llvm to llvm translate tool."));
}
