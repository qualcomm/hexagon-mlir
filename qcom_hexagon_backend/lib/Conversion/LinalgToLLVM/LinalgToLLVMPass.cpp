//===- LinalgToLLVMPass.cpp - Linalg to LLVM  conversion       ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization and lowering of MLIR IR to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "linalg-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGTOLLVM
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct LinalgToLLVMPass : public ::impl::LinalgToLLVMBase<LinalgToLLVMPass> {
public:
  explicit LinalgToLLVMPass(const LinalgToLLVMOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, async::AsyncDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect, ttx::TTXDialect,
                    tptr::HexagonTPtrDialect, hexagonmem::HexagonMemDialect,
                    hexkl::HexKLDialect, quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    setTargetTriple(moduleOp);
    setDataLayout(moduleOp);

    auto setIndexBitwidth = [&](auto passOption) {
      passOption.indexBitwidth = 32;
      return passOption;
    };

    auto setFusion = [&](auto passOption) {
      passOption.fusionAllowRecompute = fusionAllowRecompute;
      passOption.fusionDoMultiUse = fusionDoMultiUse;
      return passOption;
    };

    auto setExtendPack = [&](auto passOption) {
      passOption.upperFrontier = extendPackUpperFrontier;
      passOption.lowerFrontier = extendPackLowerFrontier;
      passOption.parallelsOnly = extendPackParallelsOnly;
      return passOption;
    };

    auto setVTCMTiling = [&](auto passOption) {
      passOption.tileSizes = tileSizes;
      return passOption;
    };

    auto setuseInterchangeVector = [&](auto passOption) {
      passOption.useInterchangeVector = useInterchangeVector;
      return passOption;
    };

    auto setOpSlicingFactor = [&](auto passOption) {
      passOption.slicingFactor = slicingFactor;
      return passOption;
    };

    auto setsplitTilingRange = [&](auto passOption) {
      passOption.splitTilingRange = splitTilingRange;
      return passOption;
    };

    auto setenableSplitReduction = [&](auto passOption) {
      passOption.enableSplitReduction = enableSplitReduction;
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

    auto setLWP = [&](auto passOption) {
      passOption.disableLWPLoop = disableLWPLoop;
      passOption.LWPloopDepth = LWPloopDepth;
      return passOption;
    };

    // Set ConvTiling flags
    auto setConvTiling = [&](auto passOption) {
      passOption.convTilingFactor = convTilingFactor;
      passOption.convTileHeightDim = convTileHeightDim;
      passOption.convTileOCDim = convTileOCDim;
      return passOption;
    };

    PassManager pm(&getContext(), moduleOp.getOperationName());

    // RequestCWrappersPass adds an attribute to a function if it has a return
    // value which would generate a c-wrapper function during the
    // FuncToLLVMPass. See here for more information:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
    //
    // For example given the following function definition:
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32>
    //
    // After the RequestCWrappersPass it is converted to,
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32> attributes {llvm.emit_c_interface}
    //
    // After FuncToLLVMPass we get an additional function,
    // llvm.func @_mlir_ciface_foobar(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
    // attributes {llvm.emit_c_interface} {
    //   %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //   %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x
    //   i64>, array<2 x i64>)>
    //   ...
    //   %8 = llvm.call @foobar(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr,
    //   !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //
    //   llvm.store %8, %arg0 :
    //   !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>,
    //   !llvm.ptr
    //
    //   llvm.return
    // }
    //
    if (doesFuncReturnValue(moduleOp))
      pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());

    pm.addNestedPass<func::FuncOp>(createLowerTTXPass());
    pm.addPass(createLowerLibdevicePass());
    pm.addNestedPass<func::FuncOp>(createLowerTPtrPass());
    pm.addNestedPass<func::FuncOp>(createHexagonLowerTmTensorPass());
    pm.addNestedPass<func::FuncOp>(createReduceContractionRankPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (puntBuffer)
      pm.addNestedPass<func::FuncOp>(createHexagonPuntBufferPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung allocs

    // Quantization related passes in this block
    // Lower quant.qcast and quant.dcast ops to arith dialect
    pm.addNestedPass<func::FuncOp>(quant::createLowerQuantOps());
    // Convert arith ops to linalg elementwise ops
    pm.addPass(createConvertElementwiseToLinalgPass());
    // Remove quant.scast ops
    pm.addPass(createCSEPass());

    if (enableHexKL)
      pm.addNestedPass<func::FuncOp>(createMatmulToHexKLPass());

    if (enableConvTiling) {
      pm.addNestedPass<func::FuncOp>(
          createConvTilingPass(setConvTiling(ConvTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    pm.addNestedPass<func::FuncOp>(createConvertLayoutPass());
    pm.addNestedPass<func::FuncOp>(createScheduleMatmulForHVXPass());
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizePass());

    if (returnValueOptimization)
      pm.addNestedPass<func::FuncOp>(createHexagonRVOPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung re-interprets
    pm.addPass(createCSEPass());

    if (enableSCFThreading) {
      assert(!enableMultiThreading && !enableVTCMTiling &&
             "currently scf-threading can be enabled only if"
             " linalg multi-threading and vtcm tiling are off");
      pm.addNestedPass<func::FuncOp>(createFormSCFThreadsPass());
    }

    if (fusion)
      pm.addNestedPass<func::FuncOp>(
          createHexagonFusionPass(setFusion(HexagonFusionOptions{})));
    pm.addPass(createEraseUnusedLinalgOperands());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (enableSlicing)
      pm.addPass(createHexagonSlicingPass(
          setOpSlicingFactor(HexagonSlicingOptions{})));

    pm.addNestedPass<func::FuncOp>(createDecomposeTensorConcatPass());

    if (enableVTCMTiling) {
      pm.addNestedPass<func::FuncOp>(
          createVTCMTilingPass(setVTCMTiling(VTCMTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    if (enableMultiThreading) {
      pm.addNestedPass<func::FuncOp>(
          createFormVirtualThreadsPass(FormVirtualThreadsOptions{}));
    }

    pm.addPass(removeMLProgramPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    if (enableVectorization) {
      pm.addPass(
          createHexagonTilingPass(setsplitTilingRange(setuseInterchangeVector(
              setenableSplitReduction(HexagonTilingOptions{})))));
    }

    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(
        createSmallExponentToMultiplyPass(SmallExponentToMultiplyOptions{}));
    // ===== STEP 1: HOIST SCALAR OPS =====
    // Run before vectorization to expose scalar invariants
    pm.addNestedPass<func::FuncOp>(createHoistScalarOpsPass());
    pm.addPass(createEraseUnusedLinalgOperands());
    pm.addPass(createCSEPass());

    // ===== STEP 1.5: LOOP INVARIANT CODE MOTION =====
    // Move hoisted scalars further up the loop nest
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run LICM again after canonicalization to catch newly exposed
    // opportunities
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());

    // ===== STEP 2: VECTORIZATION =====
    // Vectorizer now sees cleaner IR with hoisted scalars
    if (enableVectorization) {
      pm.addPass(createHexagonVectorizationPass());
    }
    pm.addPass(createRewriteUBPoisonToZeroPass());
    pm.addPass(createHexagonVectorLoweringPass());
    pm.addPass(createCanonicalizerPass());

    if (addFastMath) {
      pm.addPass(createHexagonAddFastMathPass());
      pm.addNestedPass<func::FuncOp>(createFoldMulFByZeroPass());
      pm.addPass(createCanonicalizerPass());
    }
    pm.addPass(memref::createResolveShapedTypeResultDimsPass());

    if (enableBufferization) {
      pm.addPass(bufferization::createEmptyTensorEliminationPass());
      mlir::bufferization::OneShotBufferizePassOptions passOpts;
      passOpts.bufferizeFunctionBoundaries = true;
      passOpts.allowReturnAllocsFromLoops = true;
      pm.addPass(bufferization::createOneShotBufferizePass(passOpts));
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());

      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS1Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          bufferization::createBufferLoopHoistingPass());

      pm.addNestedPass<func::FuncOp>(createCopyCanonicalizationPass());
      pm.addPass(createCanonicalizerPass());

      bufferization::buildBufferDeallocationPipeline(
          pm, bufferization::BufferDeallocationPipelineOptions{});

      pm.addPass(createCSEPass());
      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS2Pass());
      }

      pm.addNestedPass<func::FuncOp>(createConvertZeroSizeMemrefPass());
      pm.addPass(createConvertBufferizationToMemRefPass());
    }

    if (enableConvertToHexagonmem)
      pm.addNestedPass<func::FuncOp>(createConvertToHexagonmemPass());

    // Decompose hexkl.matmul to micro ops
    if (enableHexKL)
      pm.addNestedPass<func::FuncOp>(createDecomposeHexKLMatmulPass());

    // Lower linalg ops with library_call attribute set to custom fns.
    pm.addPass(createHexagonReplaceWithLibraryCallsPass());
    if (enableHexagonmemCopyToDMA)
      pm.addNestedPass<func::FuncOp>(createHexmemCpyToDMAPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertLinalgToLoopsPass());

    pm.addNestedPass<func::FuncOp>(createFormAsyncThreadsPass());
    pm.addPass(createAsyncFuncToAsyncRuntimePass());
    pm.addPass(createAsyncToAsyncRuntimePass());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

    if (enableLWP)
      pm.addNestedPass<func::FuncOp>(
          createHexagonLWPPass(setLWP(mlir::hexagon::HexagonLWPPassOptions{})));

    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createExpandMathOpsPass());

    if (expandBoolVec)
      pm.addNestedPass<func::FuncOp>(createExpandBoolVecPass());

    pm.addPass(createFastInversePass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass(
        setIndexBitwidth(ConvertIndexToLLVMPassOptions{})));

    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass(ConvertFuncToLLVMPassOptions{}));

    pm.addPass(hexagon::createDMAToLLVMPass());
    pm.addPass(hexagonmem::createHexagonMemToLLVMPass());
    pm.addPass(hexkl::createHexKLToLLVMPass());

    if (enableCollapseAddressSpace) {
      pm.addPass(createCollapseAddressSpacePass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    }

    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (enableHexagonRoutines)
      pm.addPass(createHexagonLLVMEnableHexagonRoutinesPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLinalgToLLVMPass(const LinalgToLLVMOptions &options) {
  return std::make_unique<LinalgToLLVMPass>(options);
}
