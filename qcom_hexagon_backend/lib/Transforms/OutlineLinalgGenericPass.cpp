//===- OutlineLinalgGenericPass.cpp - Outline linalg.generic ops ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Outline every linalg.generic op into its own private func.func and replace
// the original op with a func.call. The outlined functions are marked with the
// LLVM `noinline` passthrough attribute so that the downstream LLVM -O3
// optimizer (which includes the inliner and runs in both translate-to-llvmir
// and translate-to-obj) does NOT re-merge them back into a single huge
// function. Keeping the functions separate lets the LLVM backend (llc codegen)
// and the Hexagon linker process many small functions instead of one
// monolithic one, reducing llvm->obj and obj->so compile time.
//
// The op's tensor operands (inputs + inits) and any values captured from above
// (e.g. hoisted scalars / constants used inside the region) become function
// arguments; the op's results become the function results.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_OUTLINELINALGGENERIC
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static bool isGeneratedOutlineHelper(func::FuncOp func) {
  StringRef name = func.getSymName();
  return name.starts_with("__outlined_generic_") ||
         name.starts_with("__merged_outlined_generic_");
}

struct OutlineLinalgGenericPass
    : public ::impl::OutlineLinalgGenericBase<OutlineLinalgGenericPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // Collect generics first; we mutate the IR while outlining, and we must not
    // recurse into the functions we create (which contain a generic each).
    // Entry points can be private by the time this pass runs. Skip only
    // helpers emitted by this pass or FindIdenticalLinalgGenericsPass.
    SmallVector<linalg::GenericOp> generics;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      if (func.isExternal() || isGeneratedOutlineHelper(func))
        continue;
      func.walk([&](linalg::GenericOp op) { generics.push_back(op); });
    }

    // `noinline` passthrough attr forwarded by FuncToLLVM -> llvm.func and then
    // lowered to the LLVM `noinline` function attribute, surviving -O3
    // inlining.
    auto noInlineAttr = ArrayAttr::get(ctx, {StringAttr::get(ctx, "noinline")});

    OpBuilder moduleBuilder(moduleOp.getBodyRegion());
    Block *moduleBody = moduleOp.getBody();
    SymbolTable symbolTable(moduleOp);

    unsigned counter = 0;
    unsigned outlinedCount = 0;
    for (linalg::GenericOp op : generics) {
      Location loc = op.getLoc();

      // Operands of the op (input + init/output tensors).
      SmallVector<Value> callArgs(op->getOperands().begin(),
                                  op->getOperands().end());

      // Values referenced inside the region but defined above it (scalars,
      // constants, etc.) must be threaded in as extra function arguments.
      SetVector<Value> captures;
      getUsedValuesDefinedAbove(op->getRegions(), captures);
      for (Value c : captures)
        callArgs.push_back(c);

      SmallVector<Type> argTypes;
      for (Value v : callArgs)
        argTypes.push_back(v.getType());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());

      auto fnType = moduleBuilder.getFunctionType(argTypes, resultTypes);
      std::string fnName;
      do {
        fnName = ("__outlined_generic_" + Twine(counter++)).str();
      } while (symbolTable.lookup(fnName));

      // Create the outlined function before the module terminator.
      OpBuilder::InsertionGuard guard(moduleBuilder);
      moduleBuilder.setInsertionPoint(moduleBody, std::prev(moduleBody->end()));
      auto outlinedFunc =
          func::FuncOp::create(moduleBuilder, loc, fnName, fnType);
      outlinedFunc.setPrivate();
      outlinedFunc->setAttr("passthrough", noInlineAttr);

      // Build the body: map each call argument to the corresponding block arg,
      // clone the generic, and return its results.
      Block *entry = outlinedFunc.addEntryBlock();
      OpBuilder bodyBuilder(entry, entry->end());
      IRMapping mapping;
      for (auto [val, arg] : llvm::zip(callArgs, entry->getArguments()))
        mapping.map(val, arg);

      Operation *cloned = bodyBuilder.clone(*op.getOperation(), mapping);
      func::ReturnOp::create(bodyBuilder, loc, cloned->getResults());

      // Replace the original op with a call to the outlined function.
      OpBuilder callBuilder(op);
      auto call =
          func::CallOp::create(callBuilder, loc, outlinedFunc, callArgs);
      op->replaceAllUsesWith(call.getResults());
      op->erase();
      ++outlinedCount;
    }

    llvm::errs() << "[outline-linalg-generic] outlined " << outlinedCount
                 << " linalg.generic op(s)\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createOutlineLinalgGenericPass() {
  return std::make_unique<OutlineLinalgGenericPass>();
}
