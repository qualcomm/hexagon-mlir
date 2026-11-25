# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_library(Transforms
  DoubleBufferGenericS1.cpp
  DoubleBufferGenericS2.cpp
  Transforms.cpp
  OptionsParsing.cpp
  LowerLibdevicePass.cpp
  LowerTritonPrintPass.cpp
  LegalizeFP8CastsToLLVMCallPass.cpp
  LegalizeLateFP8CastsToLLVMCallPass.cpp
  AddFastMathPass.cpp
  BatchMatmulToSCF.cpp
  UseHexagonRoutines.cpp
  PuntBufferPass.cpp
  RVO.cpp
  ScheduleMatmulForHVXPass.cpp
  GeneralizePass.cpp
  ReplaceWithLibraryCallsPass.cpp
  VectorLoweringPass.cpp
  DecomposeTensorConcatPass.cpp
  MatmulToConvPass.cpp
  RelaxFP32MatmulToFP16Pass.cpp
  MatmulToHexKLPass.cpp
  CopyCanonicalizationPass.cpp
  CopyCanonicalizationAllocPatterns.cpp
  HoistLoopInvariantAllocPass.cpp
  MaterializeStridedLoadsPass.cpp
  DecomposeHexKLMatmulPass.cpp
  DecomposeHexKLMatmulToDaemonPass.cpp
  InjectHMXThreadIdArgPass.cpp
  InjectHMXVtcmBaseArgPass.cpp
  InjectHMXQueueArgPass.cpp
  ConvTilingPass.cpp
  ConvertLayoutPass.cpp
  ExpandMathOpsPass.cpp
  FastExpApproxPass.cpp
  RemoveMLProgram.cpp
  ExpandBoolVec.cpp
  EraseUnusedLinalgOperands.cpp
  FastInversePass.cpp
  BroadcastedReductionToButterflyPass.cpp
  ConvertZeroSizeMemrefPass.cpp
  ConvertToHexagonmem.cpp
  LowerTPtr.cpp
  FinalizeMemRefAndCorePtrToLLVMConversion.cpp
  SeedLayoutConversions.cpp
  ForceHVXCrouton.cpp
  SmallExponentToMultiplyPass.cpp
  LinalgUtils.cpp
  PackUnpackUtils.cpp
  PreprocessTiledConv2D.cpp
  InsertScratchArgPass.cpp
  MemoryOffsetsPass.cpp
  HexagonLWPInstrumentation.cpp
  LowerTmTensor.cpp
  LowerTTX.cpp
  ReduceContractionRankPass.cpp
  FoldCastsIntoMatmul.cpp
  HoistScalarOps.cpp
  FoldMulFByZeroPass.cpp
  FoldResourceTransposePass.cpp
  LowerHexKLMatmulToMacroPass.cpp
  PreprocessWeightsForHMXPass.cpp
  FoldPackUnpackConstantsPass.cpp
  EliminateRedundantUnpackPackPass.cpp
  OutlineLinalgGenericPass.cpp
  HexagonDeallocAfterLastUsePass.cpp
  DivToMulOptimizationPass.cpp
  SCFLoopUnrollPass.cpp
  HexagonUnrollAndJamPass.cpp
  ConvertStaticAllocToAlloca.cpp
  SpecializeTranscendentalsPass.cpp
  GuardedOptimizeAllocationLivenessPass.cpp
  VectorAccumulatorForReductionPass.cpp

  DEPENDS
  TransformsPassIncGen
  HexagonBackendTableGen
  TritonTableGen


  LINK_COMPONENTS
  Core


  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationTransforms
  MLIRControlFlowInterfaces
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrToLLVM
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRShardDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRSupport
  MLIRVectorDialect
)

