//===----------- AffineToLLVM.cpp - Affine to LLVM Passes -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization and lowering of MLIR IR to LLVM.
// It assumes that linalg is the ingress IR but a lot of transformations are
// done at the affine dialect level.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/AffineToLLVM/AffineToLLVM.h"
#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "affine-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using mlir::func::FuncOp;
using namespace hexagon;

#define GEN_PASS_DEF_AFFINETOLLVM
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

namespace {

struct AffineToLLVMPass : public ::impl::AffineToLLVMBase<AffineToLLVMPass> {
public:
  explicit AffineToLLVMPass(const AffineToLLVMOptions &options)
      : Base(options) {}

  // Add depdendent dialects
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect, hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override {
    llvm::outs() << "Starting linalg-affine-llvm lowering ... \n";
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    setTargetTriple(moduleOp);
    setDataLayout(moduleOp);

    auto setIndexBitwidth = [&](auto passOption) {
      passOption.indexBitwidth = 32;
      return passOption;
    };

    auto setAllowReturnAllocs = [&](auto passOption) {
      passOption.allowReturnAllocsFromLoops = true;
      return passOption;
    };

    auto setBufferizeFunctionBoundaries = [&](auto passOption) {
      passOption.bufferizeFunctionBoundaries = true;
      return passOption;
    };

    PassManager pm(&getContext(), moduleOp.getOperationName());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Rewrite named linalg ops into generic ops.
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());

    pm.addPass(createLinalgFoldUnitExtentDimsPass());

    // One-Shot-Bufferize and related de-allocation.
    pm.addPass(bufferization::createEmptyTensorEliminationPass());
    mlir::bufferization::OneShotBufferizePassOptions passOpts;
    passOpts.bufferizeFunctionBoundaries = true;
    passOpts.allowReturnAllocsFromLoops = true;
    pm.addPass(bufferization::createOneShotBufferizePass(passOpts));

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());
    bufferization::buildBufferDeallocationPipeline(
        pm, bufferization::BufferDeallocationPipelineOptions{});

    // Lower Linalg to Affine Loops
    pm.addPass(createConvertLinalgToAffineLoopsPass());

    // Tile affine loops
    pm.addPass(createAffineTilingPass(AffineTilingOptions{}));

    // Vectorize
    pm.addNestedPass<func::FuncOp>(createAffineVectorizePass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createAffineToLLVMPass(const AffineToLLVMOptions &options) {
  return std::make_unique<AffineToLLVMPass>(options);
}
