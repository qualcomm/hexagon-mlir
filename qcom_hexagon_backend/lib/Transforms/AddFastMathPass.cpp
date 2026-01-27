//===- AddFastMathPass.cpp - enable fast-math for arith::FOps -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass adds fast-math to ops that are either LLVM-Ops or implement
// fast math interface.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-add-fast-math"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONADDFASTMATH
#include "hexagon/Transforms/Passes.h.inc"

namespace {
static void addFastMathFlag(Operation *op) {
  // We could set at LLVM level but thats too late.
  LLVM::FastmathFlags llvmFMF = LLVM::FastmathFlags::fast;
  TypeSwitch<Operation *>(op)
      .Case<LLVM::FMulOp, LLVM::FAddOp, LLVM::FSubOp, LLVM::FNegOp>(
          [&](auto fOp) { fOp.setFastmathFlags(llvmFMF); });

  IRRewriter rewriter(op->getContext());
  auto fmfAttrName =
      rewriter.getStringAttr(arith::AddFOp::getFastMathAttrName());

  auto arithFMF = arith::FastMathFlags::fast;

  auto fmfOpInterface = dyn_cast<arith::ArithFastMathInterface>(op);
  if (fmfOpInterface) {
    auto attr = arith::FastMathFlagsAttr::get(op->getContext(), arithFMF);
    op->setAttr(fmfAttrName, attr);
  }
}

struct HexagonAddFastMathPass
    : public ::impl::HexagonAddFastMathBase<HexagonAddFastMathPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }
  void runOnOperation() override {
    getOperation()->walk([](Operation *op) { addFastMathFlag(op); });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonAddFastMathPass() {
  return std::make_unique<HexagonAddFastMathPass>();
}
