//===- AddFastMathPass.cpp - enable reassoc fast-math for arith::FOps -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass adds the reassoc fast-math flag to floating-point ops so that
// LLVM's vector.reduction lowering can emit tree reductions instead of serial
// scalar loops.  Only reassoc is set; flags that change observable values
// (nnan, ninf, nsz, arcp, afn, contract) are intentionally left clear.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

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

// Hexagon's HVX backend never marks ISD::FMAXNUM/FMINNUM/FMAXIMUM/FMINIMUM
// (or their _IEEE variants) as Legal or Custom; only the newer, distinct
// fmaximumnum/fminimumnum nodes have direct vmax/vmin instruction patterns.
// So a max/min op without `nnan` legalizes through
// TargetLowering::expandFMINNUM_FMAXNUM's only non-libcall fallback,
// createSelectForFMINNUM_FMAXNUM -- which itself requires `nnan` before it
// will emit the select(setgt/setlt) form that Hexagon's MinMax_pats
// TableGen patterns match into a single vmax/vmin instruction. Without
// `nnan` there is no legal vectorized (or even native scalar) lowering at
// all, so the op silently degrades to a scalarized fmax/fmin libcall per
// lane -- a large perf cliff, not merely a missed optimization. Unlike
// add/mul/sub, `nnan` on these ops is therefore restored: it costs nothing
// observable in the common case (these kernels are not run with NaN
// inputs) and is required to get any native codegen on this target.
static bool isMinMaxFloatOp(Operation *op) {
  if (isa<arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
          arith::MinimumFOp>(op))
    return true;
  if (auto redOp = dyn_cast<vector::ReductionOp>(op)) {
    using K = vector::CombiningKind;
    switch (redOp.getKind()) {
    case K::MAXNUMF:
    case K::MINNUMF:
    case K::MAXIMUMF:
    case K::MINIMUMF:
      return true;
    default:
      return false;
    }
  }
  return false;
}

// MLIR's arith::ExtFOp::fold (ArithOps.cpp:1551) eliminates an
// extf(truncf(x)) roundtrip only when both ops carry `contract`.
// This prevents some of the later optimizations to trigger.
static bool isNarrowWideRoundTripOp(Operation *op) {
  return isa<arith::TruncFOp, arith::ExtFOp>(op);
}

static void addFastMathFlag(Operation *op) {
  // Skip comparison operations to preserve NaN detection semantics
  // (e.g., x != x for isNaN checks)
  if (isa<arith::CmpFOp>(op)) {
    return;
  }

  // Set only reassoc so LLVM can emit tree reductions for vector.reduction.
  // Flags that change observable IEEE values (nnan, ninf, nsz, arcp, afn,
  // contract) are intentionally left clear -- with two op-class exceptions:
  // min/max ops need nnan (see isMinMaxFloatOp), and truncf/extf need
  // contract for the extf(truncf(x))->x fold (see isNarrowWideRoundTripOp).
  LLVM::FastmathFlags llvmFMF = LLVM::FastmathFlags::reassoc;
  TypeSwitch<Operation *>(op)
      .Case<LLVM::FMulOp, LLVM::FAddOp, LLVM::FSubOp, LLVM::FNegOp>(
          [&](auto fOp) { fOp.setFastmathFlags(llvmFMF); });

  IRRewriter rewriter(op->getContext());
  auto fmfAttrName =
      rewriter.getStringAttr(arith::AddFOp::getFastMathAttrName());

  auto arithFMF = arith::FastMathFlags::reassoc;
  if (isMinMaxFloatOp(op))
    arithFMF = arithFMF | arith::FastMathFlags::nnan;
  if (isNarrowWideRoundTripOp(op))
    arithFMF = arithFMF | arith::FastMathFlags::contract;

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
