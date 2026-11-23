//===----- Transforms.h - Linalg lowering and optimization passes
//-------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_TRANSFORMS_H
#define HEXAGON_TRANSFORMS_TRANSFORMS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
#define GEN_PASS_DECL
#include "hexagon/Transforms/Passes.h.inc"

/// Unit-attribute name used to mark an op whose float result is mathematically
/// non-negative for every element. Producers that lower a provably non-negative
/// value into an otherwise opaque op sequence (e.g. FastExpApprox replacing
/// math.exp with a polynomial / bit-trick) set this on their result op so that
/// later sign-sensitive rewrites (e.g. FastInversePass) can still treat the
/// value as >= 0. The marker is an assertion by the producer; consumers must
/// fail safe (treat its absence as "unknown sign").
inline constexpr llvm::StringLiteral kHexagonNonNegativeAttrName =
    "hexagon.nonneg";

std::unique_ptr<OperationPass<ModuleOp>>
createFinalizeMemRefAndCorePtrToLLVMConversionPass(
    const FinalizeMemRefAndCorePtrToLLVMConversionOptions &options =
        FinalizeMemRefAndCorePtrToLLVMConversionOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createCopyCanonicalizationPass();

std::unique_ptr<OperationPass<func::FuncOp>>

createHexagonHoistLoopInvariantAllocPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createHexagonHoistLoopInvariantAllocaPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createHexagonMaterializeStridedLoadsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createConvTilingPass(const ConvTilingOptions &options = ConvTilingOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createConvertLayoutPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertToHexagonmemPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertZeroSizeMemrefPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeTensorConcatPass();

std::unique_ptr<OperationPass<ModuleOp>> createEraseUnusedLinalgOperands();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createExpandBoolVecPass();

std::unique_ptr<OperationPass<func::FuncOp>> createExpandMathOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createFastExpApproxPass(FastExpApproxOptions options = FastExpApproxOptions());

std::unique_ptr<OperationPass<ModuleOp>> createFastInversePass();

std::unique_ptr<OperationPass<ModuleOp>>
createBroadcastedReductionToButterflyPass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonAddFastMathPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS1Pass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS2Pass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonLLVMEnableHexagonRoutinesPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonLowerTmTensorPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createGuardedOptimizeAllocationLivenessPass();

std::unique_ptr<OperationPass<func::FuncOp>> createHexagonLWPPass(
    const HexagonLWPPassOptions &options = HexagonLWPPassOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonPuntBufferPass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonReplaceWithLibraryCallsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexagonRVOPass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonVectorLoweringPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgGeneralizePass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerLibdevicePass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerTritonPrintPass(
    const LowerTritonPrintOptions &options = LowerTritonPrintOptions());

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeFP8CastsToLLVMCallPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeLateFP8CastsToLLVMCallPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTPtrPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTTXPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMatmulToConvPass();

std::unique_ptr<OperationPass<func::FuncOp>> createRelaxFP32MatmulToFP16Pass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMatmulToHexKLPass(
    const MatmulToHexKLOptions &options = MatmulToHexKLOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createDecomposeHexKLMatmulPass();

/// Injects a trailing memref<?xi8, 1> VTCM-base argument
/// (hexkl_hmx.vtcm_base_arg) into every func.func.  Used by the HexKL
/// micro-daemon path; must run after InjectHMXThreadIdArgPass and before
/// InjectHMXQueueArgPass.
std::unique_ptr<Pass> createInjectHMXVtcmBaseArgPass();

/// Decomposes hexkl.matmul into worker layout ops plus hexkl.micro_hmx_dispatch
/// / hexkl.micro_hmx_wait_idle calls for the dedicated-HMX-thread (daemon)
/// model.  Requires the thread-id, VTCM-base and queue args to be injected.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createDecomposeHexKLMatmulToDaemonPass();

std::unique_ptr<OperationPass<func::FuncOp>> createInsertScratchArgPass(
    const InsertScratchArgOptions &options = InsertScratchArgOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createMemoryOffsetsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createScheduleMatmulForHVXPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createSeedLayoutConversionsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createForceHVXCroutonPass();

std::unique_ptr<OperationPass<ModuleOp>> createSmallExponentToMultiplyPass(
    const SmallExponentToMultiplyOptions &options =
        SmallExponentToMultiplyOptions());
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createPreprocessTiledConv2DPass();

std::unique_ptr<OperationPass<ModuleOp>> removeMLProgramPass();

std::unique_ptr<Pass> createReduceContractionRankPass();

std::unique_ptr<Pass> createFoldCastsIntoMatmulPass();

std::unique_ptr<Pass> createHoistScalarOpsPass();

std::unique_ptr<Pass> createFoldMulFByZeroPass();

std::unique_ptr<Pass> createFoldResourceTransposePass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLowerHexKLMatmulToMacroPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPreprocessWeightsForHMXPass();

std::unique_ptr<Pass> createFoldPackUnpackConstantsPass();

std::unique_ptr<Pass> createEliminateRedundantUnpackPackPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDivToMulOptimizationPass();

std::unique_ptr<OperationPass<func::FuncOp>> createSCFLoopUnrollPass(
    const SCFLoopUnrollOptions &options = SCFLoopUnrollOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createHexagonUnrollAndJamPass(
    const HexagonUnrollAndJamOptions &options = HexagonUnrollAndJamOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createStaticAllocToAllocaPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createStaticAllocToAllocaPass(int64_t maxTotalAllocaSizeBytes);

std::unique_ptr<OperationPass<ModuleOp>> createLLVMAlignmentPropagationPass(
    const LLVMAlignmentPropagationOptions &options =
        LLVMAlignmentPropagationOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createSetAliasScopesPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createSpecializeTranscendentalsPass();

//===----------------------------------------------------------------------===//
// HMX matmul lowering passes (HexKL micro-daemon model)
//===----------------------------------------------------------------------===//

/// Injects a trailing i32 thread-ID argument (hmx.thread_id_arg) into every
/// func.func.  Must run before DecomposeHexKLMatmulToDaemonPass.
std::unique_ptr<Pass> createInjectHMXThreadIdArgPass();

/// Injects a trailing !llvm.ptr HMX queue argument (hmx.queue_arg) into every
/// func.func.  Must run after InjectHMXThreadIdArgPass and before
/// DecomposeHexKLMatmulToDaemonPass.  Combined with InjectHMXThreadIdArgPass,
/// the resulting trailing args are `(..., %tid: i32, %queue: !llvm.ptr)`.
std::unique_ptr<Pass> createInjectHMXQueueArgPass();

} // namespace hexagon
} // namespace mlir

#endif //  HEXAGON_TRANSFORMS_TRANSFORMS_H
