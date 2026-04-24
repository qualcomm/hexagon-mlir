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
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
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
#include "llvm/Transforms/IPO.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h" // for llvm::isa / mlir::isa

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

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

/// Classify argument types. We treat tensors, memrefs, and LLVM pointers as
/// "buffer-like". Everything else (integers, floats, index, vectors, etc.)
/// is "scalar-like".
// bool isBufferLike(mlir::Type t) {
//   if (mlir::isa<mlir::TensorType>(t))
//     return true;
//   if (mlir::isa<mlir::MemRefType>(t) ||
//   mlir::isa<mlir::UnrankedMemRefType>(t))
//     return true;
//   if (mlir::isa<mlir::LLVM::LLVMPointerType>(t))
//     return true;
//   return false;
// }

bool isBufferLike(mlir::Type t) {
  return llvm::isa<mlir::TensorType, mlir::BaseMemRefType,
                   mlir::LLVM::LLVMPointerType>(t);
}

/// Build a permutation that moves all buffer-like arguments to the front,
/// preserving the original relative order within buffers and within scalars.
///
/// Example: types [i32, memref<4xf32>, f32, tensor<2xf32>] =>
/// permutation [1, 3, 0, 2] (new order: memref, tensor, i32, f32).
llvm::SmallVector<unsigned>
buildBufferFirstPermutation(mlir::func::FuncOp func) {
  llvm::SmallVector<unsigned> bufferIdxs;
  llvm::SmallVector<unsigned> scalarIdxs;

  unsigned n = func.getNumArguments();
  bufferIdxs.reserve(n);
  scalarIdxs.reserve(n);

  for (unsigned i = 0; i < n; ++i) {
    mlir::Type t = func.getArgument(i).getType();
    (isBufferLike(t) ? bufferIdxs : scalarIdxs).push_back(i);
  }

  llvm::SmallVector<unsigned> perm;
  perm.reserve(n);
  perm.append(bufferIdxs.begin(), bufferIdxs.end());
  perm.append(scalarIdxs.begin(), scalarIdxs.end());
  return perm;
}

/// Reorder arguments and update call sites.
/// In-place reorder of one function's arguments:
/// - validates the permutation
/// - updates func::CallOp call sites
/// - rewrites entry block arguments
/// - updates function type and arg attributes
mlir::LogicalResult reorderFuncArgsAndCalls(mlir::ModuleOp &module,
                                            mlir::func::FuncOp func,
                                            llvm::ArrayRef<unsigned> perm) {
  unsigned n = func.getNumArguments();
  if (perm.size() != n)
    return func.emitError("Permutation size mismatch: got ")
           << perm.size() << ", expected " << n;

  // Validate permutation: values must be unique and < n
  llvm::SmallVector<unsigned> seen(n, 0);
  llvm::SmallVector<unsigned> invPerm(n); // inverse permutation
  for (unsigned newIdx = 0; newIdx < n; ++newIdx) {
    unsigned oldIdx = perm[newIdx];
    if (oldIdx >= n)
      return func.emitError("Invalid old index in permutation: ") << oldIdx;
    if (++seen[oldIdx] != 1)
      return func.emitError("Permutation must be a bijection. Index ")
             << oldIdx << " appears multiple times.";
    invPerm[oldIdx] = newIdx;
  }

  // Prepare old types
  llvm::SmallVector<mlir::Type> oldTypes(func.getArgumentTypes().begin(),
                                         func.getArgumentTypes().end());

  // New function type
  llvm::SmallVector<mlir::Type> newTypes(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx)
    newTypes[newIdx] = oldTypes[perm[newIdx]];
  auto newFuncType = mlir::FunctionType::get(func.getContext(), newTypes,
                                             func.getResultTypes());

  // Extract old arg attrs
  llvm::SmallVector<mlir::DictionaryAttr> oldArgAttrs(n);
  if (auto allAttrsOpt = func.getArgAttrs()) {
    auto allAttrs = allAttrsOpt.value();
    for (unsigned i = 0; i < n && i < allAttrs.size(); ++i)
      oldArgAttrs[i] =
          mlir::dyn_cast_or_null<mlir::DictionaryAttr>(allAttrs[i]);
  }

  // Compute new arg attrs
  llvm::SmallVector<mlir::DictionaryAttr> newArgAttrs(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx)
    newArgAttrs[newIdx] = oldArgAttrs[perm[newIdx]];

  // Update call sites
  if (auto uses = mlir::SymbolTable::getSymbolUses(func, module)) {
    for (auto &use : *uses) {
      mlir::Operation *user = use.getUser();
      if (auto call = llvm::dyn_cast<mlir::func::CallOp>(user)) {
        auto oldOperands = call.getArgOperands();
        if (oldOperands.size() != n)
          return call.emitError("Call operand count mismatch: got ")
                 << oldOperands.size() << ", expected " << n;

        llvm::SmallVector<mlir::Value> newOperands(n);
        for (unsigned newIdx = 0; newIdx < n; ++newIdx)
          newOperands[newIdx] = oldOperands[perm[newIdx]];
        call->setOperands(newOperands);
      }
    }
  }

  // External function: just update type and attrs
  if (func.isExternal()) {
    func.setType(newFuncType);
    for (unsigned i = 0; i < n; ++i)
      func.setArgAttrs(i, newArgAttrs[i]);
    return mlir::success();
  }

  // Rewrite entry block
  mlir::Block &entry = func.getBody().front();
  llvm::SmallVector<mlir::BlockArgument> oldArgs(entry.getArguments().begin(),
                                                 entry.getArguments().end());
  if (oldArgs.size() != n)
    return func.emitError("Entry block arg count mismatch: got ")
           << oldArgs.size() << ", expected " << n;

  llvm::SmallVector<mlir::Location> oldLocs;
  oldLocs.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    oldLocs.push_back(oldArgs[i].getLoc());

  llvm::SmallVector<mlir::BlockArgument> newArgs(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx) {
    unsigned oldIdx = perm[newIdx];
    newArgs[newIdx] =
        entry.addArgument(oldArgs[oldIdx].getType(), oldLocs[oldIdx]);
  }

  // Replace uses (O(n) using invPerm)
  for (unsigned oldIdx = 0; oldIdx < n; ++oldIdx) {
    unsigned newIdx = invPerm[oldIdx];
    oldArgs[oldIdx].replaceAllUsesWith(newArgs[newIdx]);
  }

  // Erase old arguments
  for (int i = static_cast<int>(n) - 1; i >= 0; --i)
    entry.eraseArgument(i);

  // Update function type and attrs
  func.setType(newFuncType);
  for (unsigned i = 0; i < n; ++i)
    func.setArgAttrs(i, newArgAttrs[i]);

  return mlir::success();
}

bool reorderFuncArgsAndCallsTensorFirst(mlir::ModuleOp &module_op,
                                        const std::string &fname) {
  bool changed = false;
  for (auto func : module_op.getOps<mlir::func::FuncOp>()) {
    if (func.getName() != fname)
      continue;
    unsigned n = func.getNumArguments();
    if (n == 0)
      continue;

    // Build buffer-first permutation.
    llvm::SmallVector<unsigned> perm = buildBufferFirstPermutation(func);

    // If already buffer-first, skip.
    // Requires: perm is a permutation of 0..n-1 (distinct, full coverage)
    bool isIdentity = llvm::is_sorted(perm) && !perm.empty() &&
                      perm.front() == 0 && perm.back() == perm.size() - 1;

    if (isIdentity)
      continue;

    if (!failed(reorderFuncArgsAndCalls(module_op, func, perm))) {
      changed = true;
    }
  }
  return changed;
}

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

  registry.insert<mlir::crouton::CroutonDialect>();
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
      mlir::Hexagon::Translate::linkRuntimeModules(
          llvmContext, current_llvm_mod, options_map);

      // Aggressive inlining to improve performance after linking runtime
      // modules Check if inlining is enabled via options_map
      mlir::hexagon::cond_run_inliner(current_llvm_mod,
                                      optionsLinalgToLLVM.enableHVXInlining);

      // Dumping of the principal module if requested
      const char *dumpFile = std::getenv("LLVM_IR_DUMP_TO_FILE");
      if (dumpFile && dumpFile[0] != '\0') {
        std::error_code EC;
        llvm::raw_fd_ostream outFile(dumpFile, EC, llvm::sys::fs::OF_Text);
        if (EC) {
          llvm::errs() << "Error opening LLVM IR file: " << EC.message()
                       << "\n";
        } else {
          current_llvm_mod->print(outFile, nullptr);
          outFile.flush();
        }
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

  mlir::Hexagon::Translate::linkRuntimeModules(llvmContext, llvmModule,
                                               options_map);

  // Aggressive inlining to improve performance after linking runtime modules
  // Check if inlining is enabled via options_map
  mlir::hexagon::LinalgToLLVMOptions optionsLinalgToLLVM;
  setLinalgToLLVMOptions(optionsLinalgToLLVM, options_map);

  mlir::hexagon::cond_run_inliner(llvmModule,
                                  optionsLinalgToLLVM.enableHVXInlining);

  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();
  return str;
}

} // namespace hexagon_backend