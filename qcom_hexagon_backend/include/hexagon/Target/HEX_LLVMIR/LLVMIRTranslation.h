//===- LLVMIRTranslation.h ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TARGET_HEXAGON_LLVMIRTRANSLATION_H
#define HEXAGON_TARGET_HEXAGON_LLVMIRTRANSLATION_H
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace hexagon {

enum class LLVMOptimizationLevel {
  NoOptimization = -1, // No optimization at all at the LLVM level
  O0 = 0,
  O1 = 1,
  O2 = 2,
  O3 = 3
};

// add external dependent libs
void addHexExternalLibs(mlir::ModuleOp &module,
                        const std::vector<std::string> &names,
                        const std::vector<std::string> &paths);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module> translateHexagonMlirLlvmToLLVMIR(
    llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
    const std::unordered_map<std::string, std::string> &arch_kwargs,
    LLVMOptimizationLevel optLevel);

// Overload for dumping object file from llvm::Module&
std::vector<char> llvm_module_to_obj_string(llvm::Module &llvmModule);

// Generate object file from mlir-llvm module.
std::vector<char>
llvm_module_to_obj_string(std::unique_ptr<llvm::Module> &llvmModule);

// Dump Hexagon assembly to file.
void dumpHexagonAssembly(llvm::Module &module, unsigned moduleId,
                         const std::string &archVersion);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_TARGET_HEXAGON_LLVMIRTRANSLATION_H
