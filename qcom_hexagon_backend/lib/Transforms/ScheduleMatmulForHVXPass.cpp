//===- ScheduleMatmulForHVXPass.cpp - Scheduling matmul ops          ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file specifies the rules to schedule matmul op and its variants.
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
#include "hexagon/Transforms/LinalgUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "schedule-matmul-for-hvx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SCHEDULEMATMULFORHVX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct ScheduleMatmulForHVXPass
    : public ::impl::ScheduleMatmulForHVXBase<ScheduleMatmulForHVXPass> {
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

FailureOr<linalg::GenericOp>
ScheduleMatmulForHVXPass::GeneralizeOp(IRRewriter &rewriter,
                                       linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::GenericOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  if (failed(generalizedOp)) {
    linalgOp->emitOpError("failed to generalize linalg named op");
    signalPassFailure();
    return failure(); // Return a failure result
  }
  return generalizedOp;
}

void ScheduleMatmulForHVXPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Optimize below Matmul variants
  //   linalg.matmul(a, transpose_b) -> linalg.matmul_transpose_b
  //   linalg.matmul(transpose_a, b) -> linalg.matmul_transpose_a
  //   linalg.batch_matmul(a, transpose_b) -> linalg.batch_matmul_transpose_b
  //   linalg.batch_matmul(transpose_a, b) -> linalg.batch_matmul_transpose_a
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::MatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_b
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp = rewriter.create<linalg::MatmulTransposeBOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{linalgOp.getDpsInputs()[0], transposeOp.getOperand(0)},
            linalgOp.getDpsInits());
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_a
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp = rewriter.create<linalg::MatmulTransposeAOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{transposeOp.getOperand(0), linalgOp.getDpsInputs()[1]},
            linalgOp.getDpsInits());
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    } else if (isa_and_nonnull<linalg::BatchMatmulOp>(
                   linalgOp.getOperation()) &&
               !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_b op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp =
            rewriter.create<linalg::BatchMatmulTransposeBOp>(
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{linalgOp.getDpsInputs()[0],
                           transposeOp.getOperand(0)},
                linalgOp.getDpsInits());
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_a op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp =
            rewriter.create<linalg::BatchMatmulTransposeAOp>(
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{transposeOp.getOperand(0),
                           linalgOp.getDpsInputs()[1]},
                linalgOp.getDpsInits());
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    }
  });

  SmallVector<linalg::LinalgOp> batchMatmulOps;
  // Collect all linalg::BatchMatmulOps that do not have the
  // "library_call" attribute
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::BatchMatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      batchMatmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : batchMatmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getBatchMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine batch matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }

  SmallVector<linalg::LinalgOp> matmulOps;
  // Collect all linalg::MatmulOps that do not have the "library_call" attribute
  funcOp.walk([&](linalg::MatmulOp linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      matmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : matmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createScheduleMatmulForHVXPass() {
  return std::make_unique<ScheduleMatmulForHVXPass>();
}
