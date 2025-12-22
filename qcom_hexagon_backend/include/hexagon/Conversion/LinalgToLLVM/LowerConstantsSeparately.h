//===- LowerConstantsSeparately.h - Separate out large constants ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
//  Separates the large constant tensors (typically conv/matmul weights) by
//  lowering them into separate shared objects (.so).
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H
#define HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H

#include <iomanip>
#include <iostream>
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h" // For divideCeil

#include "hexagon/Conversion/LinalgToLLVM/Passes.h" // For the base class LowerConstantsSeparatelyBase

#define DEBUG_TYPE "hexagon-lower-constants-separately"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERCONSTANTSSEPARATELY
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Kept only for easier debugging to print the data of a blob (from the
// dialect_resources) that's in hexadecimal format with two characters per byte
void printValueOfConstant(ArrayRef<char> originalData) {
  for (char c : originalData) {
    // Print each byte as a two-digit hexadecimal number.
    // std::hex sets the output format to hexadecimal.
    // std::setw(2) ensures each byte is printed with at least 2 characters.
    // std::setfill('0') pads with '0' if the byte is less than 2 characters.
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)(unsigned char)c;
  }
  std::cout << std::endl;
}

// This class implements the main logic of the constant splitting, which will
// move constant tensors to (potentially many) new separate modules, which will
// all be compiled and linked individually.
struct LowerConstantsSeparatelyPass
    : public ::impl::LowerConstantsSeparatelyBase<
          LowerConstantsSeparatelyPass> {

  // Maximum size above which a module is considered "full" and a new one will
  // be created (can happen several times, leading to several const-only mods)
  const int maximumBytesPerSharedObject = 5e8; // 500,000,000 bytes = 0.5 GB

  // Utility function to compute the memory footprint (in bytes) of a constant
  // tensor op, which will be used to know when to create a new mlir module.
  // If passed an operation that's not a valid constant tensor (containing
  // int or float values) it returns 0.
  int estimateConstantTensorMemoryFootprint(mlir::LLVM::GlobalOp op) {
    // Ensure the global is marked constant
    if (!op.getConstant()) {
      return 0;
    }

    mlir::Type type = op.getType();
    mlir::Type previousType;
    int64_t totalElements = 1;

    // Unwrap nested LLVM array types to get shape and element type
    while (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
      totalElements *= arrayType.getNumElements();
      previousType = type;
      type = arrayType.getElementType();
      // Break if type doesn't change (safety against infinite loop)
      if (type == previousType) {
        return 0;
      }
    }

    // Now `type` should be the element type (int or float)
    if (!type.isIntOrFloat()) {
      std::cerr << "Element type is not int or float." << std::endl;
      return 0;
    }

    int64_t elementBitWidth = type.getIntOrFloatBitWidth();
    int64_t elementByteSize = llvm::divideCeil(elementBitWidth, 8);
    int64_t totalSize = totalElements * elementByteSize;

    return totalSize;
  }

  // Main function of the pass, which goes through the module, looking for
  // constants to be moved to (potentially) many new mlir modules. Does so
  // while making sure that each produced module stays under the limit
  // defined by `maximumBytesPerSharedObject`. Takes care not only of
  // moving the constants, but also of moving the associated entried defined
  // in the dialect_resources
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext &newContext = *(mod.getContext());

    // Create a DialectRegistry and register the necessary extensions
    mlir::DialectRegistry newRegistry;
    mlir::registerAllExtensions(newRegistry);

    // Appends the registry to the new context
    newContext.appendDialectRegistry(newRegistry);

    // Create a new MLIR module in which we'll be putting the current op
    DBG("[LowerConstantsSeparately] Creating a new module for constants only"
        << "\n");
    mlir::ModuleOp newMod =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
    // The size of the new module will be the sum of the weights (in bytes) of
    // the constant tensors it contains. It is 0 for now since it's empty.
    int newModSize = 0;

    // Collection to store the keys of the dense resources needed, which we will
    // find as we discover the constants definitions in the original module
    std::vector<std::string> resourceKeys;

    // Walking the original MLIR module, trying to find global constants to move
    // to the new MLIR module we just made
    mod.walk([&](mlir::Operation *op) {
      // If it's a GlobalOp...
      if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
        // And if it's a constant...
        if (globalOp.getConstant()) {
          // 1 - Copy the constant definition to the new module being created
          // ------------------------------------------------------------------------
          // Then we clone the global constant
          mlir::LLVM::GlobalOp clonedOp = globalOp.clone();
          // We set its linkage attribute to External, to make sure it will be
          // accessible from other .o
          auto linkageAttrClonedOp =
              LLVM::LinkageAttr::get(&newContext, LLVM::Linkage::External);
          clonedOp.setLinkageAttr(linkageAttrClonedOp);

          // Compute the size (in bytes) needed by the current constant
          int currentOpSize = estimateConstantTensorMemoryFootprint(globalOp);

          // If the current op can't fit in the current module for constants
          if (newModSize + currentOpSize > maximumBytesPerSharedObject) {
            // Push the current mod into the collection of produced modules
            producedModules.push_back(newMod);
            // Create a new MLIR module in which we'll be putting the current op
            DBG("[LowerConstantsSeparately] Creating another new "
                "module for constants only"
                << "\n");
            newMod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
            // It's size is 0 for now
            newModSize = 0;
          } else {
            // The current module has enough space left, so we'll be using it
            newModSize += currentOpSize;
          }

          // And we insert the cloned operation (constant definition) into the
          // new module
          newMod.push_back(clonedOp);

          // Extract the key from the dense_resource attribute and add it to our
          // collection of keys that are defined (and which will therefore need
          // the relevant meta-data copied)
          if (auto denseResourceAttr =
                  mlir::dyn_cast<mlir::DenseResourceElementsAttr>(
                      globalOp->getAttr("value"))) {
            std::string key = denseResourceAttr.getRawHandle().getKey().str();
            resourceKeys.push_back(key);
          }

          // 2 - Replace the constant by an extern in the main module (for the
          // code)
          // ------------------------------------------------------------------------
          OpBuilder builder(globalOp.getContext());
          builder.setInsertionPoint(globalOp);

          // Create the necessary attributes
          auto typeAttr = TypeAttr::get(globalOp.getType());
          auto symNameAttr = builder.getStringAttr(globalOp.getSymName());
          auto linkageAttrExternGlobalOp = LLVM::LinkageAttr::get(
              builder.getContext(), LLVM::Linkage::External);

          // Create a new extern GlobalOp
          auto externGlobalOp = builder.create<LLVM::GlobalOp>(
              globalOp.getLoc(), /*resultTypes=*/TypeRange(), typeAttr,
              /*constant=*/nullptr, symNameAttr, linkageAttrExternGlobalOp,
              /*dso_local=*/nullptr, /*thread_local=*/nullptr,
              /*value=*/nullptr, /*alignment=*/nullptr, /*addr_space=*/nullptr,
              /*unnamed_addr=*/nullptr, /*section=*/nullptr, /*comdat=*/nullptr,
              /*dbg_expr=*/nullptr, /*visibility=*/nullptr);

          // Replace the original GlobalOp that was the constant definition with
          // the new "extern" GlobalOp, which references a constant from the
          // outside
          globalOp.getOperation()->replaceAllUsesWith(
              externGlobalOp.getOperation());
          globalOp.erase();
        }
      }
    });

    // Ensure the new module has the needed resources (containing the definition
    // of the constants it uses)
    auto *resourceInterface =
        mod->getContext()
            ->getLoadedDialect<mlir::BuiltinDialect>()
            ->getRegisteredInterface<
                mlir::ResourceBlobManagerDialectInterface>();
    auto *newResourceInterface =
        newMod->getContext()
            ->getLoadedDialect<mlir::BuiltinDialect>()
            ->getRegisteredInterface<
                mlir::ResourceBlobManagerDialectInterface>();

    if (resourceInterface && newResourceInterface) {
      mlir::DialectResourceBlobManager &resourceMap =
          resourceInterface->getBlobManager();
      mlir::DialectResourceBlobManager &newResourceMap =
          newResourceInterface->getBlobManager();

      // For each resource being defined in the new module, move the associated
      // blob from the original "resource blob manager" to the new one
      for (const auto &key : resourceKeys) {
        DialectResourceBlobManager::BlobEntry *blobEntry =
            resourceMap.lookup(key);
        if (blobEntry) {
          // Moving the binary blob
          DialectResourceBlobManager::BlobEntry &returned =
              newResourceMap.insert(key, std::move(*(blobEntry->getBlob())));
        }
      }
      // Copy module-level attributes
      newMod->setAttrs(mod->getAttrs());

    } else {
      std::cerr << "Error: ResourceBlobManagerDialectInterface not found!"
                << std::endl;
    }

    // Push the final module for constants (the non-full one)
    producedModules.push_back(newMod);
  }

  // Method to access the newly produced modules
  std::vector<ModuleOp> getProducedModules() const { return producedModules; }

private:
  // Modules produced by the pass, which contain only constants (moved
  // from the original module)
  std::vector<ModuleOp> producedModules;

}; // end of class

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLowerConstantsSeparatelyPass();

#endif // HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H
