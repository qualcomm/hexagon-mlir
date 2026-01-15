//===--- AffineVectorize.cpp - Affine Vectorization Pass ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This file implements vectorization of affine loops.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

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

#include "hexagon/Conversion/AffineToLLVM/AffineToLLVM.h"

#define DEBUG_TYPE "affine-vectorize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using namespace mlir::affine;

#define GEN_PASS_DEF_AFFINEVECTORIZE
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

namespace {

struct AffineVectorizePass
    : public ::impl::AffineVectorizeBase<AffineVectorizePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    std::vector<SmallVector<AffineForOp, 2>> loops;
    gatherLoops(getOperation(), loops);
    if (loops.empty())
      return;

    // vectorize the deepest loops only for now.
    // Note: deepest is subset of innermost
    auto &deepLoops = loops[loops.size() - 1];

    for (unsigned i = 0; i < deepLoops.size(); ++i) {
      AffineForOp loop = deepLoops[i];
      if (!isLoopParallel(loop, nullptr))
        continue;

      VectorizationStrategy strategy;
      strategy.vectorSizes.push_back(32 /*vectorization factor*/);
      strategy.loopToVectorDim[loop] = 0;
      std::vector<SmallVector<AffineForOp, 2>> loopsToVectorize;
      loopsToVectorize.push_back({loop});
      (void)vectorizeAffineLoopNest(loopsToVectorize, strategy);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createAffineVectorizePass() {
  return std::make_unique<AffineVectorizePass>();
}
