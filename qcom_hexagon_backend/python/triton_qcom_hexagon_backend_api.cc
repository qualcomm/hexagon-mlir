//===-- triton_qcom_hexagon_backend_api.cc --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "triton_qcom_hexagon_backend_api.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexKL/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "hexagon/Target/HEX_LLVMIR/LLVMIRTranslation.h"
#include "hexagon/Target/Linalg_MLLVMIR/MLLVMIRTranslation.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // added this for the loadDialect

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include <regex>

#include "LinkRuntimeModules.h"

#define DEBUG_TYPE "triton-qcom-hexagon-backend-api"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Wrapper over std::runtime_error to signal LLVM IR parsing errors
void fail(const std::string &message) { throw std::runtime_error(message); }

std::unique_ptr<llvm::Module>
fixAlignedAllocTypes(llvm::LLVMContext &context, llvm::Module *originalModule,
                     const std::string &outputFilePath) {

  // Dump LLVM IR to string
  std::string llvm_ir_str;
  llvm::raw_string_ostream rso(llvm_ir_str);
  originalModule->print(rso, nullptr);
  rso.flush();

  // Regular expressions to replace aligned_alloc declarations and calls
  // of type (i64, i64) with (i32, i32), which was needed to support
  // the async runtime.
  std::regex decl_pattern(R"(declare\s+ptr\s+@aligned_alloc\(i64,\s*i64\))");
  std::string decl_replacement = "declare ptr @aligned_alloc(i32, i32)";

  std::regex call_pattern(R"(@aligned_alloc\(i64\s+(\d+),\s+i64\s+(\d+)\))");
  std::string call_replacement = "@aligned_alloc(i32 $1, i32 $2)";

  std::string modified_ir_str =
      std::regex_replace(llvm_ir_str, decl_pattern, decl_replacement);
  modified_ir_str =
      std::regex_replace(modified_ir_str, call_pattern, call_replacement);

  // Optionally write to file
  if (!outputFilePath.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream outFile(outputFilePath, EC, llvm::sys::fs::OF_Text);
    if (EC)
      llvm::errs() << "Error writing to file: " << EC.message() << "\n";
    else
      // Parse modified IR back into llvm::Module
      outFile << modified_ir_str;
  }

  // Parse modified IR back into llvm::Module
  llvm::SMDiagnostic err;
  auto modifiedModule = llvm::parseIR(
      llvm::MemoryBufferRef(modified_ir_str, "modified_module"), err, context);

  if (!modifiedModule)
    fail("Failed to parse modified LLVM IR after aligned_alloc fix.");

  return modifiedModule;
}

namespace hexagon_backend {
// Extract the return types (rank and dtype) of a function given its name.
std::vector<std::pair<int, mlir::Type>>
getReturnList(mlir::ModuleOp module_op, const std::string &fName) {
  std::vector<std::pair<int, mlir::Type>> resultList;

  for (auto funcOp : module_op.getOps<mlir::func::FuncOp>()) {
    if (funcOp.getName() != fName)
      continue;
    for (auto resultType : funcOp.getResultTypes()) {
      int rank;
      mlir::Type dtype;

      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        rank = tensorType.getRank();
        dtype = tensorType.getElementType();
      } else {
        // Assuming if it's not RankedTensor, it's a scalar value.
        rank = 0;
        dtype = resultType;
      }
      resultList.push_back(std::make_pair(rank, dtype));
    }
  }
  // Return the list of return types.
  return resultList;
}

std::string extractSingleFuncName(mlir::ModuleOp &module_op) {
  std::string module_func_name;
  std::vector<std::string> vector_func_names;
  for (auto func : module_op.getOps<mlir::func::FuncOp>())
    vector_func_names.push_back(func.getName().str());
  if (vector_func_names.size() != 1)
    fail("Expected exactly one function in the module.");

  module_func_name = vector_func_names[0];
  return module_func_name;
}

void loadDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);   // TODO: restrict
  mlir::registerAllExtensions(registry); // TODO: restrict

  registry.insert<mlir::hexagonmem::HexagonMemDialect>();
  registry.insert<mlir::hexkl::HexKLDialect>();
  registry.insert<mlir::tm_tensor::TmTensorDialect>();
  registry.insert<mlir::ttx::TTXDialect>();
  registry.insert<mlir::tptr::HexagonTPtrDialect>();

  // Register all external models.
  mlir::hexkl::registerBufferizableOpInterfaceExternalModels(registry);

  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::triton::TritonDialect>();
  context.loadAllAvailableDialects();
}

mlir::ModuleOp parseMlirFromFile(const std::string &path,
                                 mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  if (!module)
    fail("Parse MLIR file failed.");
  return module->clone();
}

mlir::ModuleOp parseMlirFromString(const std::string &src,
                                   mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(src, &context);
  if (!module)
    fail("Parse MLIR string failed.");
  return module->clone();
}

std::vector<std::vector<char>> translateLinalgToObj(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map) {
  // The collection of object codes (each seen as a sequence of
  // bytes/char) that will be produced
  std::vector<std::vector<char>> mods_object_codes_as_bytes;

  // Needed to know if we should lower the constants separately or not
  mlir::hexagon::LinalgToLLVMOptions optionsLinalgToLLVM;
  // sets optionsLinalgToLLVM using options_map
  setLinalgToLLVMOptions(optionsLinalgToLLVM, options_map);

  std::vector<mlir::ModuleOp> mods;
  // Normal mode, where constants stay in the main module
  // ----------------------------------------------------
  if (!optionsLinalgToLLVM.lowerConstantsInSeparateSharedObjects) {
    // Old (default) mode where the Linalg/MLIR module gets lowered
    // to a single LLVM/MLIR module
    auto mod =
        ::mlir::hexagon::translateLinalgToLLVMMLIR(linalg_module, options_map);
    if (!mod)
      fail("Failed to convert Triton Linalg to LLVM MLIR.");
    mods.push_back(mod);
  } else {
    // New mode where constants are lowered separately into their own .o
    // and .so
    DBG("STEP 0 - Calling translateLinalgToMultipleLLVMMLIRModules"
        << "\n");
    mods = ::mlir::hexagon::translateLinalgToMultipleLLVMMLIRModules(
        linalg_module, options_map);
    if (mods.empty())
      fail("Error in the production of multiple modules: no modules produced");
  }

  DBG("There are a total of " << mods.size()
                              << " modules, including the main (code) one"
                              << "\n");
  // Iterating through each LLVM/MLIR module that has been produced
  int module_id = -1; // current module being treated
  for (mlir::ModuleOp current_mlir_mod : mods) {
    module_id++;
    if (!current_mlir_mod)
      fail("Failed to convert Triton Linalg to LLVM/MLIR for the current "
           "module" +
           std::to_string(module_id));

    llvm::LLVMContext llvmContext;
    DBG("[Module " << module_id
                   << "] - STEP 1 - Calling translateHexagonMlirLlvmToLLVMIR"
                   << "\n");
    // If it's the main module (for the code), use the optLevel 3,
    // otherwise no opt to avoid the modules containing only constants of
    // being completely elided
    mlir::hexagon::LLVMOptimizationLevel opt_level_translation =
        (module_id == 0) ? mlir::hexagon::LLVMOptimizationLevel::O3
                         : mlir::hexagon::LLVMOptimizationLevel::NoOptimization;
    // LLVM/MLIR module to LLVM-IR
    std::unique_ptr<llvm::Module> current_llvm_mod =
        ::mlir::hexagon::translateHexagonMlirLlvmToLLVMIR(
            &llvmContext, current_mlir_mod, options_map, opt_level_translation);

    if (!current_llvm_mod)
      fail("Failed to translate LLVM/MLIR to LLVM-IR for the current module" +
           std::to_string(module_id));

    // Only if it's the principal module (=code), link the runtime modules
    // with it
    if (module_id == 0) {
      DBG("[Module " << module_id
                     << "] - Extra step - Calling linkRuntimeModules for the "
                        "principal module"
                     << "\n");
      mlir::Hexagon::Translate::linkRuntimeModules(llvmContext,
                                                   current_llvm_mod);

      // Dumping of the principal module if requested
      if (mlir::hexagon::isEnvTrue("LLVM_IR_ENABLE_DUMP")) {
        std::string mod_string;
        std::unique_ptr<llvm::raw_string_ostream> ir_ss(
            new llvm::raw_string_ostream(mod_string));
        current_llvm_mod->print(*ir_ss, nullptr);
        llvm::outs() << "// -----// LLVM IR Dump //----- //\n"
                     << mod_string << "\n";
      }
    }

    DBG("[Module " << module_id
                   << "] - STEP 2 - Calling llvm_module_to_obj_string"
                   << "\n");
    auto replacedAlignedModule =
        fixAlignedAllocTypes(llvmContext, current_llvm_mod.get(), "");

    if (!replacedAlignedModule)
      fail("Failed to fix aligned_alloc types in LLVM module for the current "
           "module" +
           std::to_string(module_id));

    if (mlir::hexagon::isEnvTrue("HEXAGON_ASM_DUMP") ||
        mlir::hexagon::isEnvTrue("HEXAGON_ASM_TO_OBJ")) {
      const char *archEnv = std::getenv("HEXAGON_ARCH_VERSION");
      if (!archEnv) {
        llvm::errs() << "Warning: HEXAGON_ASM_DUMP is set but "
                        "HEXAGON_ARCH_VERSION is not set. Skipping "
                        "assembly dump.\n";
      } else {
        std::string archVersion = std::string(archEnv);
        mlir::hexagon::dumpHexagonAssembly(*replacedAlignedModule, module_id,
                                           archVersion);
      }
    }

    std::vector<char> object_code_as_bytes =
        mlir::hexagon::llvm_module_to_obj_string(replacedAlignedModule);
    // For now, the collection returned has only one object code
    mods_object_codes_as_bytes.push_back(object_code_as_bytes);
  }

  return mods_object_codes_as_bytes;
}

std::string translateLinalgToLLVMIR(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map) {
  auto mod =
      ::mlir::hexagon::translateLinalgToLLVMMLIR(linalg_module, options_map);
  if (!mod)
    fail("Failed to convert Triton Linalg to LLVM MLIR.");

  // llvm mlir module to llvm ir
  llvm::LLVMContext llvmContext;

  auto llvmModule = ::mlir::hexagon::translateHexagonMlirLlvmToLLVMIR(
      &llvmContext, mod, options_map, mlir::hexagon::LLVMOptimizationLevel::O3);

  if (!llvmModule)
    fail("Failed to translate TritonHexagon to LLVM IR.");

  mlir::Hexagon::Translate::linkRuntimeModules(llvmContext, llvmModule);

  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();
  return str;
}

} // namespace hexagon_backend
