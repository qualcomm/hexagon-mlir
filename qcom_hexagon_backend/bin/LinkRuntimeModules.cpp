//===- LinkRuntimeModules.cpp ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Implementation of functions to link runtime modules with the kernel.
//
//===----------------------------------------------------------------------===//

#include "LinkRuntimeModules.h"
#include "llvm/Linker/Linker.h"
#include <cassert>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <map>
#include <queue>
#include <set>

#define INIT_MODULE(NAME)                                                      \
  extern const unsigned char hexagon_runtime_module_##NAME[];                  \
  extern const unsigned int hexagon_runtime_module_##NAME##_len;               \
  static std::unique_ptr<llvm::Module> get_module_##NAME(                      \
      llvm::LLVMContext &context) {                                            \
    auto bitcode = hexagon_runtime_module_##NAME;                              \
    auto bitcode_size = hexagon_runtime_module_##NAME##_len;                   \
    return get_module(context, bitcode, bitcode_size);                         \
  }

#define ADD_MODULE_CAND(NAME, CONTEXT, EXTERNAL_MODULE, MODULE_CANDIDATES)     \
  otherModule = get_module_##NAME(CONTEXT);                                    \
  addModToCandidates(EXTERNAL_MODULE, MODULE_CANDIDATES);

void condLinkModules(
    llvm::Linker &linker, const std::set<llvm::Module *> &modulesToLink,
    std::map<llvm::Module *, std::unique_ptr<llvm::Module>> &moduleCandidates) {
  for (auto otherMod : modulesToLink) {
    if (linker.linkInModule(std::move(moduleCandidates[otherMod])))
      llvm::report_fatal_error("Error linking module\n");
  }
}

static std::unique_ptr<llvm::Module> get_module(llvm::LLVMContext &context,
                                                const unsigned char *bitcode,
                                                unsigned int bitcode_size) {
  // Create a StringRef from the bitcode data
  llvm::StringRef bitcodeRef(reinterpret_cast<const char *>(bitcode),
                             bitcode_size);
  // Create a MemoryBuffer from the StringRef
  auto buffer = llvm::MemoryBuffer::getMemBuffer(bitcodeRef, "", false);
  // Parse the bitcode file
  auto moduleOrErr = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  if (!moduleOrErr)
    llvm::report_fatal_error(llvm::Twine("Error parsing bitcode file: ") +
                             llvm::toString(moduleOrErr.takeError()));
  return std::move(*moduleOrErr);
}
// Initialize all the runtime modules here.
INIT_MODULE(IntrinsicsHVX);
INIT_MODULE(VTCMPool);
INIT_MODULE(HexagonAPI);
INIT_MODULE(HexagonBuffer);
INIT_MODULE(HexagonBufferAlias);
INIT_MODULE(HexagonCAPI);
INIT_MODULE(HexKLAPI);
INIT_MODULE(RuntimeDMA);
INIT_MODULE(UserDMA);

// collectFuncsInMod collects the functions in the passed module. We distinguish
// between functions that are declarations and the ones that are definitions
// This function helps find the declarations in modules that we know we will use
// (e.g. the module of the kernel in Triton) and the definitions in external
// modules.
std::set<std::string>
collectFuncsInMod(const std::unique_ptr<llvm::Module> &mod, bool shouldBeDecl) {
  std::set<std::string> funcsInMod;

  for (llvm::Function &func : *mod) {
    bool isDecl = func.isDeclaration();
    if ((shouldBeDecl && isDecl) || ((!shouldBeDecl) && (!isDecl))) {
      std::string functionName = func.getName().str();
      funcsInMod.insert(functionName);
    }
  }
  return funcsInMod;
}

// Function that produces two maps that will be useful both for getting the
// direct dependencies for the main module we're trying to link against as well
// as the indirect dependencies definingMod is the only one we need to find
// direct dependencies since we need to only parse the declarations in the main
// module and then find out who defines these declarations. declsInMod is useful
// for finding indirect dependencies. For every module we figure that it needs
// to be linked, we can analyze which are its declarations. We use declarations
// to detect defining modules that were not linked yet.
void buildFunctionModuleMaps(
    std::unordered_map<std::string, llvm::Module *> &definingMod,
    std::unordered_map<llvm::Module *, std::set<std::string>> &declsInMod,
    const std::map<llvm::Module *, std::unique_ptr<llvm::Module>>
        &moduleCandidates) {
  for (auto it = moduleCandidates.begin(); it != moduleCandidates.end(); ++it) {
    for (auto funcDecl : collectFuncsInMod(it->second, true))
      declsInMod[it->first].insert(funcDecl);
    for (auto funcDef : collectFuncsInMod(it->second, false))
      definingMod[funcDef] = it->first;
  }
}

// Breadth-first-search algorithm that 1) detects the declarations in modules
// that we decided to link against (modulesToLink) using the declsInMod map, 2)
// Finds out which modules define these declarations with definingMod and 3)
// updates modulesToLink with the newly found dependent modules
void getDependentCandidates(
    std::set<llvm::Module *> &modulesToLink,
    const std::unordered_map<std::string, llvm::Module *> &definingMod,
    const std::unordered_map<llvm::Module *, std::set<std::string>>
        &declsInMod) {

  // newModulesToLink contains the newly discovered modules that need to be
  // linked in a breadth-first-search traversal (bfs). newModulesToLink is
  // different from modulesToLink since the latter contains both already
  // identified modules for linking as well as the ones in newModulesToLink
  // modulesToLink is a set since we need to run quick searches of module
  // pointers on it whereas newModulesToLink only needs traversals
  std::queue<llvm::Module *> newModulesToLink;
  for (auto elem : modulesToLink)
    newModulesToLink.push(elem);
  while (not newModulesToLink.empty()) {
    auto m = newModulesToLink.front();
    newModulesToLink.pop();
    auto modDeclsIt = declsInMod.find(m);
    // if a module doesn't contain any declarations, there are
    // no definitions we need to search for for that module
    if (modDeclsIt == declsInMod.end())
      continue;
    for (auto funcName : modDeclsIt->second) {
      auto it = definingMod.find(funcName);
      // functions belonging to standard libraries won't be found in external
      // modules
      if (it == definingMod.end())
        continue;
      llvm::Module *modDefiningFunc = it->second;
      if (modulesToLink.find(modDefiningFunc) == modulesToLink.end()) {
        newModulesToLink.push(modDefiningFunc);
        modulesToLink.insert(modDefiningFunc);
      }
    }
  }
}

void addModToCandidates(
    std::unique_ptr<llvm::Module> &otherModule,
    std::map<llvm::Module *, std::unique_ptr<llvm::Module>> &moduleCandidates) {
  llvm::Module *ptr = otherModule.get();
  moduleCandidates[ptr] = std::move(otherModule);
}

void mlir::Hexagon::Translate::linkRuntimeModules(
    llvm::LLVMContext &ctx, std::unique_ptr<llvm::Module> &module) {

  // We need a map from the module raw pointers to the unique pointers since
  // in our algorithm for detecting dependent modules we use two different
  // maps that handle module pointers and their declarations and definitions.
  // We use the raw pointers since we need to have module pointers in the two
  // maps at the same time (impossible with unique_ptr). And we also need to
  // keep track of the unique pointers since at the end of the day we need
  // to call linker.linkInModule which only accepts a unique pointer
  std::map<llvm::Module *, std::unique_ptr<llvm::Module>> moduleCandidates;

  std::unique_ptr<llvm::Module> otherModule;
  ADD_MODULE_CAND(IntrinsicsHVX, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(VTCMPool, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(HexagonAPI, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(HexagonBuffer, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(HexagonBufferAlias, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(HexagonCAPI, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(HexKLAPI, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(RuntimeDMA, ctx, otherModule, moduleCandidates);
  ADD_MODULE_CAND(UserDMA, ctx, otherModule, moduleCandidates);

  llvm::Linker linker(*module);
  llvm::Module *mainModule = module.get();
  // module ownership passed to moduleCandidates
  // it will be passed back to module by end of this function
  addModToCandidates(module, moduleCandidates);
  std::unordered_map<std::string, llvm::Module *> definingMod;
  std::unordered_map<llvm::Module *, std::set<std::string>> declsInMod;
  buildFunctionModuleMaps(definingMod, declsInMod, moduleCandidates);

  // we add module to the modules link just temporarily to
  // detect its external module dependencies
  std::set<llvm::Module *> modulesToLink = {mainModule};
  getDependentCandidates(modulesToLink, definingMod, declsInMod);
  modulesToLink.erase(mainModule);

  condLinkModules(linker, modulesToLink, moduleCandidates);
  // unique pointer to main module is still used afterwards
  module = std::move(moduleCandidates[mainModule]);
}
