//===- ReplaceWithLibraryCallsPass.cpp - replace funcs with lib calls -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file replaces named linalg ops with "library_call" attribute set to
// external function calls. The pass is almost a replica of the
// LinalgToStandard pass, but it converts only linalg ops with the library_call
// attribute set instead of all linalg ops.
// The pass needs to run before GeneralizePass to preseve named linalg ops.
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "hexagon-replace-with-library-calls"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;

#define GEN_PASS_DEF_HEXAGONREPLACEWITHLIBRARYCALLS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

static SmallVector<Value, 4>
createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc,
                                      ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = dyn_cast<MemRefType>(op.getType());
    if (!memrefType) {
      res.push_back(op);
      continue;
    }
    Value cast =
        b.create<memref::CastOp>(loc, makeStridedLayoutDynamic(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = dyn_cast<MemRefType>(type))
      result.push_back(makeStridedLayoutDynamic(memrefType));
    else
      result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
static FailureOr<FlatSymbolRefAttr>
getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter) {
  auto linalgOp = dyn_cast<LinalgOp>(op);
  assert(linalgOp && "Expecting a linalgOp here");
  auto fnName = linalgOp->getAttrOfType<StringAttr>("library_call");
  if (!fnName)
    return rewriter.notifyMatchFailure(op, "No library call defined for op");

  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));
  if (op->getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Library call for linalg operation can be generated only for ops that "
        "have void return types");
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();
  return fnNameAttr;
}
class ReplaceWithLibraryCallsPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
    if (failed(libraryCallName))
      return failure();

    // TODO: Add support for more complex library call signatures that include
    // indices or captured values.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, libraryCallName->getValue(), TypeRange(),
        createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(),
                                              op->getOperands()));
    return success();
  }
};

void populateReplaceWithLibraryCallsPattern(RewritePatternSet &patterns) {
  patterns.insert<ReplaceWithLibraryCallsPattern>(patterns.getContext());
}

struct HexagonReplaceWithLibraryCallsPass
    : public ::impl::HexagonReplaceWithLibraryCallsBase<
          HexagonReplaceWithLibraryCallsPass> {
public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    populateReplaceWithLibraryCallsPattern(patterns);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  /// Register the dialects that must be loaded in the context before this pass.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonReplaceWithLibraryCallsPass() {
  return std::make_unique<HexagonReplaceWithLibraryCallsPass>();
}
