//===- Generalize.cpp - linalg named op to generic ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering most of the named to linalg generic.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "linalg-generalize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGGENERALIZE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LinalgGeneralizePass
    : public ::impl::LinalgGeneralizeBase<LinalgGeneralizePass> {
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

FailureOr<linalg::GenericOp>
LinalgGeneralizePass::GeneralizeOp(IRRewriter &rewriter,
                                   linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::GenericOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  return generalizedOp;
}

void LinalgGeneralizePass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::LinalgOp> matmulOps;
  // Collect all linalg::MatmulOps that do not have the "library_call" attribute
  funcOp.walk([&](linalg::MatmulOp linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      matmulOps.push_back(linalgOp);
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto linalgOp : matmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize linalg named op");
      return signalPassFailure();
    }
  }

  SmallVector<linalg::LinalgOp> namedOps;
  // Collect all linalg named ops
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<
            linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp, linalg::CeilOp,
            linalg::CopyOp, linalg::DivOp, linalg::DivUnsignedOp,
            linalg::ElementwiseOp, linalg::ExpOp, linalg::FloorOp,
            linalg::LogOp, linalg::MapOp, linalg::MaxOp, linalg::MulOp,
            linalg::NegFOp, linalg::ReduceOp, linalg::MatmulTransposeBOp,
            linalg::BatchMatmulTransposeBOp,
            // Fix for Flash-Attention : Don't generalize transpose
            linalg::TransposeOp, linalg::SubOp>(linalgOp.getOperation())) {
      namedOps.push_back(linalgOp);
    }
  });

  for (auto linalgOp : namedOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize linalg named op");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createLinalgGeneralizePass() {
  return std::make_unique<LinalgGeneralizePass>();
}
