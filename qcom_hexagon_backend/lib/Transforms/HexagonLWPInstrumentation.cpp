//===- HexagonLWPInstrumentation.cpp - light weight profiler --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// A light weight profiler.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define DEBUG_TYPE "hexagon-lwp-instrumentation"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace std::literals;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::LLVM;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONLWPPASS
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct HexagonLWPPass : public ::impl::HexagonLWPPassBase<HexagonLWPPass> {

  explicit HexagonLWPPass(const HexagonLWPPassOptions &options)
      : HexagonLWPPassBase(options) {}

  // Compute loop depth (distance to outermost loop)
  static int getLoopDepth(Operation *op) {
    int depth = 0;
    while (auto parent = dyn_cast_or_null<scf::ForOp>(op->getParentOp())) {
      ++depth;
      op = parent;
    }
    return depth;
  }

  // Check if a loop has at least one sibling scf.for loop
  static bool hasSiblingLoop(scf::ForOp forOp) {
    Operation *parentOp = forOp->getParentOp();
    if (!parentOp || parentOp->getNumRegions() == 0)
      return false;

    auto &block = parentOp->getRegion(0).front();
    for (auto &op : block) {
      if (&op != forOp.getOperation() && isa<scf::ForOp>(op))
        return true;
    }
    return false;
  }

  // Dump the location and ops info into a file so they can be postprocessed.
  void printLines(const std::vector<int> &lines, int loopId,
                  const std::vector<std::string> &opNames) const {
    static bool fileInitialized = false;
    const std::string filePath = "/tmp/lwp_infodump.txt";

    // Remove the stale file for fresh start.
    if (!fileInitialized) {
      if (llvm::sys::fs::exists(filePath)) {
        llvm::sys::fs::remove(filePath);
      }
      fileInitialized = true;
    }

    std::error_code EC;
    llvm::raw_fd_ostream fileStream(filePath, EC, llvm::sys::fs::OF_Append);

    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      return;
    }

    fileStream << "Location ";
    for (size_t i = 0; i < lines.size(); ++i) {
      fileStream << lines[i];
      if (i != lines.size() - 1)
        fileStream << ", ";
    }

    fileStream << " corresponds to ID " << loopId << " | Collected ops: ";

    if (opNames.empty()) {
      fileStream << "\n";
    } else {
      for (size_t i = 0; i < opNames.size(); ++i) {
        fileStream << opNames[i];
        if (i != opNames.size() - 1)
          fileStream << ", ";
      }
      fileStream << "\n";
    }

    fileStream.flush();
  }

  // Get location info in a list.
  static std::vector<int> collectLoc(Location loc) {
    std::vector<int> lines;
    // FusedLoc can have nested FusedLoc.
    // Recursively get all the relevant locations in lines list.
    std::function<void(Location)> recurse = [&](Location loc) {
      if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
        for (Location subLoc : fusedLoc.getLocations())
          recurse(subLoc);
      } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
        lines.push_back(fileLoc.getLine());
      }
    };

    recurse(loc);
    return lines;
  }

  // Collect ops of interest.
  std::vector<std::string> collectOps(scf::ForOp forOp) {
    std::vector<std::string> opNames;
    forOp.getBody()->walk([&](Operation *op) {
      // Filter out non-trivial operations
      auto opName = op->getName().getStringRef();

      if (opName == "memref.subview" || opName == "memref.load" ||
          opName == "memref.store" || opName == "vector.insertelement" ||
          opName == "vector.transfer_read" ||
          opName == "vector.transfer_write" || opName == "vector.broadcast" ||
          opName == "vector.extractelement" || opName == "affine.apply" ||
          opName == "cf.assert" || opName == "scf.yield" ||
          opName == "scf.for") {
        return;
      }

      if (auto reductionOp = dyn_cast<mlir::vector::ReductionOp>(op)) {
        auto kind =
            reductionOp.getKind(); // This returns a vector::CombiningKind enum
        std::string formattedName = "vector.reduction<" +
                                    vector::stringifyCombiningKind(kind).str() +
                                    ">";
        opNames.push_back(formattedName);
      } else {
        opNames.push_back(opName.str());
      }
    });

    return opNames;
  }

  void runOnOperation() override {
    auto func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(module.getContext());

    auto i8Ty = IntegerType::get(ctx, 8);
    auto strType = LLVM::LLVMArrayType::get(i8Ty, 12);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i8PtrTy = LLVMPointerType::get(ctx);

    static int increment_loopId = 1;

    // Retrieve an existing LLVM global variable named "handler_name" or
    // create it to store the string "lwp_handler".
    auto getOrCreateHandlerGlobal = [&]() -> LLVM::GlobalOp {
      auto global = module.lookupSymbol<LLVM::GlobalOp>("handler_name");
      if (global)
        return global;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto str = builder.getStringAttr("lwp_handler\0"s);

      return builder.create<LLVM::GlobalOp>(
          module.getLoc(), TypeAttr::get(strType), UnitAttr::get(ctx),
          builder.getStringAttr("handler_name"),
          LLVM::LinkageAttr::get(ctx, LLVM::Linkage::Internal), UnitAttr(),
          UnitAttr(), UnitAttr(), str, IntegerAttr(),
          builder.getI32IntegerAttr(0), nullptr, StringAttr(), SymbolRefAttr(),
          ArrayAttr(), LLVM::VisibilityAttr());
    };

    // Retrieve or create the LLVM function 'llvm.hexagon.instrprof.custom'.
    auto getOrCreateInstrFunc = [&]() {
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
              "llvm.hexagon.instrprof.custom")) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        auto funcTy = LLVM::LLVMFunctionType::get(
            voidTy, ArrayRef<Type>{i8PtrTy, i32Ty}, false);
        auto func = builder.create<LLVM::LLVMFuncOp>(
            module.getLoc(), "llvm.hexagon.instrprof.custom", funcTy);
      }
    };

    // Create the call to the LLVM intrinsic
    auto emitInstrumentationCall = [&](OpBuilder builder, Location loc,
                                       auto gep, Value id) {
      auto symbolRef =
          FlatSymbolRefAttr::get(ctx, "llvm.hexagon.instrprof.custom");
      auto instr = builder.create<LLVM::CallOp>(loc, TypeRange{}, symbolRef,
                                                ValueRange{gep, id});
    };

    // Get or create pointer to the string data "lwp_handler".
    LLVM::GEPOp globalGEP;
    auto getOrCreateGlobalGEP = [&](OpBuilder builder,
                                    Location loc) -> LLVM::GEPOp {
      if (globalGEP)
        return globalGEP;

      // Get the global variable "handler_name".
      auto global = getOrCreateHandlerGlobal();
      auto addr = builder.create<LLVM::AddressOfOp>(loc, global);
      auto zero = builder.create<LLVM::ConstantOp>(
          loc, i32Ty, builder.getI32IntegerAttr(0));
      globalGEP = builder.create<LLVM::GEPOp>(loc, i8PtrTy, strType, addr,
                                              ArrayRef<Value>{zero, zero});
      return globalGEP;
    };

    getOrCreateInstrFunc();

    // === Instrument Functions ===
    {
      Location loc = func.getLoc();
      std::vector<int> lines = collectLoc(loc);
      printLines(lines, increment_loopId, {});

      OpBuilder builder(func.getBody());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&func.getBody().front());

      auto gep = getOrCreateGlobalGEP(builder, loc);
      auto id = builder.create<LLVM::ConstantOp>(
          loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

      emitInstrumentationCall(builder, loc, gep, id);

      func.walk([&](func::ReturnOp retOp) {
        builder.setInsertionPoint(retOp);
        emitInstrumentationCall(builder, loc, gep, id);
      });
    }

    if (!disableLWPLoop) {
      // === Instrument Loops ===
      func.walk([&](scf::ForOp forOp) {
        int depth = getLoopDepth(forOp);

        bool shouldInstrument =
            (depth == 0) || (depth <= LWPloopDepth && hasSiblingLoop(forOp));

        if (!shouldInstrument)
          return;

        Location loc = forOp.getLoc();
        OpBuilder builder(forOp);

        std::vector<std::string> opNames = collectOps(forOp);
        std::vector<int> lines = collectLoc(loc);

        printLines(lines, increment_loopId, opNames);

        auto gep = getOrCreateGlobalGEP(builder, loc);
        auto id = builder.create<LLVM::ConstantOp>(
            loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(forOp);
          emitInstrumentationCall(builder, loc, gep, id);
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(forOp);
          emitInstrumentationCall(builder, loc, gep, id);
        }
      });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonLWPPass(const HexagonLWPPassOptions &options) {
  return std::make_unique<HexagonLWPPass>(options);
}
