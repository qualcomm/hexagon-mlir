//===- CollapseAddressSpace.cpp: lower memref ops to hexagonmem     ------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Replaces instances of !llvm.ptr<1> with bare ptr. This is needed especially
// for llvm.structs that wrap memref which bridge two different address spaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Threading.h"
#include <memory>
#include <mutex>
#include <optional>

#define DEBUG_TYPE "collapse-address-space"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_COLLAPSEADDRESSSPACE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct CollapseAddressSpacePass
    : public ::impl::CollapseAddressSpaceBase<CollapseAddressSpacePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<hexagonmem::HexagonMemDialect>();
  }
  void runOnOperation() override;
};

Type getCollapsedType(Type type, llvm::DenseMap<Type, Type> &visited) {

  if (auto it = visited.find(type); it != visited.end())
    return it->second; // return cached result

  if (!(isa<LLVM::LLVMStructType>(type) || isa<LLVM::LLVMPointerType>(type) ||
        isa<MemRefType>(type)))
    return type; // return unchanged

  // Recursive collapse.
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<LLVM::LLVMStructType>([&](auto structTy) {
        MLIRContext *ctx = structTy.getContext();
        visited[structTy] = structTy; // cache before recursion

        SmallVector<Type> memberTys;
        for (Type elemTy : structTy.getBody()) {
          Type collapsed = getCollapsedType(elemTy, visited);
          memberTys.push_back(collapsed);
        }

        auto newStruct = LLVM::LLVMStructType::getLiteral(ctx, memberTys);
        visited[structTy] = newStruct; // update cache
        return newStruct;
      })
      .Case<LLVM::LLVMPointerType>([&](auto ptrTy) {
        MLIRContext *ctx = ptrTy.getContext();
        return LLVM::LLVMPointerType::get(ctx, DEFAULT_DDR_ADDRESS_SPACE);
      })
      .Case<MemRefType>([&](auto memrefTy) {
        MLIRContext *ctx = memrefTy.getContext();
        Type elemTy = getCollapsedType(memrefTy.getElementType(), visited);
        Attribute newSpace = IntegerAttr::get(IntegerType::get(ctx, 64),
                                              DEFAULT_DDR_ADDRESS_SPACE);
        return MemRefType::get(memrefTy.getShape(), elemTy,
                               memrefTy.getLayout(), newSpace);
      })
      .Default([](Type ty) -> Type { return ty; });
}

void CollapseAddressSpacePass::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
    LLVM::LLVMFunctionType oldFuncType = funcOp.getFunctionType();
    llvm::DenseMap<Type, Type> visited; // single map cache reused below.

    // Build collapsed input and result types
    SmallVector<Type> newInputs;
    bool changed = false;
    for (Type inputType : oldFuncType.getParams()) {
      Type collapsed = getCollapsedType(inputType, visited);
      newInputs.push_back(collapsed);
      changed |= (collapsed != inputType);
    }

    Type oldResultType = oldFuncType.getReturnType();
    Type newResultType = getCollapsedType(oldResultType, visited);
    changed |= (newResultType != oldResultType);

    auto updateBlockTypes = [&](Region &region) {
      for (Block &block : region.getBlocks()) {
        // === Update block arguments ===
        for (BlockArgument arg : block.getArguments())
          arg.setType(getCollapsedType(arg.getType(), visited));

        // === Update operation result types ===
        block.walk([&](Operation *op) {
          for (unsigned i = 0; i < op->getNumResults(); ++i)
            op->getResult(i).setType(
                getCollapsedType(op->getResult(i).getType(), visited));
        });
      }
    };

    if (changed) {
      // Create a new function with the updated type
      auto newFuncType = LLVM::LLVMFunctionType::get(newResultType, newInputs,
                                                     oldFuncType.isVarArg());

      IRRewriter rewriter(funcOp.getContext());
      rewriter.setInsertionPoint(
          funcOp); // Ensure new function is inserted before the old one

      auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
          funcOp.getLoc(), funcOp.getName(), newFuncType);
      newFuncOp.setVisibility(funcOp.getVisibility());

      // Copy the passthrough attribute from the old function into the new one
      auto passthroughAttr = funcOp->getAttr("passthrough");
      if (passthroughAttr)
        newFuncOp->setAttr("passthrough", passthroughAttr);

      // Move the body from the old function to the new one
      if (!funcOp.getBody().empty()) {
        newFuncOp.getBody().takeBody(funcOp.getBody());
        updateBlockTypes(newFuncOp.getBody());
      }

      // Replace the old function with the new function
      rewriter.replaceOp(funcOp, newFuncOp);
    } else {
      updateBlockTypes(funcOp.getBody());
    }
  });
}
} // namespace

std::unique_ptr<Pass> hexagon::createCollapseAddressSpacePass() {
  return std::make_unique<CollapseAddressSpacePass>();
}
