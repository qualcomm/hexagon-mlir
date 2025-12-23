//===- ExpandBoolVecPass.cpp - implementation of expand bool vector ------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion so that `uitofp from bool to any-fp type`
// is replaced with select between two constant vectors. This is possible as
// i1 has just two possible values, and select is an elementwise operation.
// This replacement is needed as hexagon llvm isel cannot currently uitofp
// bool type. Therefore,
// e.g.
// ```mlir
//    %r = arith.uitofp %x : vector<5xi1> to vector<5xf16>
// ```
//
// will be re-written as :
// ```mlir
//    %r = arith.select %x, %cons_ones, %const_zeros
//                      : vector<5xi1>, vector<5xf16>
// ```
//
// which will lower to llvm-ir as something like:
// ```llvm
//    %r = select <5 x i1> %x, <5 x half>
//            <half 0xH3C00, half 0xH3C00, half 0xH3C00,
//             half 0xH3C00, half 0xH3C00>,
//            <5 x half> zeroinitializer
//
// Note: one needs to pay attention to floating-point semantics encoding
//       e.g. 1.0f is not auto-converted to half value in APFLoat.
//       We need to get it from llvm::fltsemenantics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-expand-bool-vec"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_EXPANDBOOLVEC
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct ExpandBoolVecPass : public ::impl::ExpandBoolVecBase<ExpandBoolVecPass> {
  void runOnOperation() override;
};

/// returns fltSemantics singleton pointer based on element-type.
static const llvm::fltSemantics *getSemantics(ShapedType shapedTy) {
  const llvm::fltSemantics *Semantics = nullptr;
  ;
  const auto elType = shapedTy.getElementType();

  if (elType.isF16()) {
    Semantics = &APFloat::IEEEhalf();
  } else if (elType.isF32()) {
    Semantics = &APFloat::IEEEsingle();
    ;
  } else if (elType.isF64()) {
    Semantics = &APFloat::IEEEdouble();
    ;
  } else
    llvm_unreachable("unhandled element type");
  return Semantics;
}

void ExpandBoolVecPass::runOnOperation() {

  getOperation().walk([&](arith::UIToFPOp op) {
    auto inType = dyn_cast<VectorType>(op.getIn().getType());
    if (!inType || !inType.getElementType().isSignlessInteger(1))
      return;

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointAfterValue(op);
    auto loc = op.getLoc();

    auto shapedTy = cast<ShapedType>(op.getType());
    const llvm::fltSemantics &Semantics = *getSemantics(shapedTy);

    auto zero = APFloat::getZero(Semantics);
    auto zeros = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(shapedTy, zero));

    auto one = APFloat::getOne(Semantics);
    auto ones = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(shapedTy, one));
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, op.getIn(), ones, zeros);
  });
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createExpandBoolVecPass() {
  return std::make_unique<ExpandBoolVecPass>();
}
