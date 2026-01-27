//===- Vectorization.cpp - Implementation of Vectorization Pass  ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of inner loop to vector form so that vector
// to llvm pass can convert to vectorized llvm-ir.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#define DEBUG_TYPE "hexagon-vectorize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONVECTORIZATION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

static LogicalResult vectorizeLinalgOp(linalg::LinalgOp op) {
  auto fnName = op->getAttrOfType<StringAttr>("library_call");
  if (fnName) {
    DBG("-> skipping vectorization. Op will be replaced with a library call.");
    return failure();
  }

  IRRewriter rewriter(op.getContext());

  auto numLoops = op.getNumLoops();
  if (numLoops < 1) {
    return failure();
  }
  auto innerLoopDim = op.getStaticLoopRanges()[numLoops - 1];

  auto dataTileSize = computeDataTileSize(op);
  if (!dataTileSize ||
      !perfectlyVectorizable(dataTileSize.value(), innerLoopDim)) {
    DBG("-> skipping vectorization. data tile size and loop mismatch");
    return failure();
  }

  SmallVector<int64_t> vecSizes(numLoops, 1);
  SmallVector<bool> scalableDims(numLoops, false);
  vecSizes[numLoops - 1] = innerLoopDim;

  FailureOr<mlir::linalg::VectorizationResult> vectorResults =
      linalg::vectorize(rewriter, op, vecSizes, scalableDims);
  if (failed(vectorResults))
    return failure();
  // Replace the original op with the vectorized op.
  rewriter.replaceOp(op, vectorResults->replacements);
  return success();
}

struct HexagonVectorizationPass
    : public ::impl::HexagonVectorizationBase<HexagonVectorizationPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG("vectorization candidate: " << op << "\n");
      if (succeeded(vectorizeLinalgOp(op))) {
        DBG(" -> vectorization succeeded.\n");
      } else {
        DBG(" -> vectorization failed.\n");
      }
      return WalkResult::advance();
    });
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonVectorizationPass() {
  return std::make_unique<HexagonVectorizationPass>();
}
