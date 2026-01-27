//===- VectorLoweringPass.cpp -  Vector Lowering  Pass   ------------------===//
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
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-vector-lowering"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONVECTORLOWERING
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static void doVectorLowering(ModuleOp moduleOp) {
  auto vectorTransposeLowering = vector::VectorTransposeLowering::Shuffle1D;
  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;

  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering);

  RewritePatternSet patterns(moduleOp.getContext());
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vectorTransformOptions.vectorContractLowering,
      /*benefit=*/1,
      /*disableOuterProductLowering=*/false);
  vector::populateScalarVectorTransferLoweringPatterns(
      patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorMultiReductionLowering);
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
  FrozenRewritePatternSet frozenPatterns = std::move(patterns);

  moduleOp.walk([&](func::FuncOp funcOp) {
    (void)applyPatternsGreedily(funcOp, frozenPatterns);
  });
}

struct HexagonVectorLoweringPass
    : public ::impl::HexagonVectorLoweringBase<HexagonVectorLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        func::FuncDialect>(); // , arith::ArithDialect, math::MathDialect,
                              // linalg::LinalgDialect, affine::AffineDialect,
                              // scf::SCFDialect, tensor::TensorDialect,
    //  bufferization::BufferizationDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {

    auto moduleOp = getOperation();
    doVectorLowering(moduleOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonVectorLoweringPass() {
  return std::make_unique<HexagonVectorLoweringPass>();
}
