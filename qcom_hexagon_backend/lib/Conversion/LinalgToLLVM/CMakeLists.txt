# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_conversion_library(LinalgToLLVM
  Common.cpp
  ExtendPackUnpackFrontier.cpp
  FormSCFThreads.cpp
  FlattenZeroLoopLinalgPass.cpp
  FormVirtualThreadsPass.cpp
  FormCompositeVirtualThreadsPass.cpp
  FormAsyncThreadsPass.cpp
  LinalgToLLVM.cpp
  LinalgToLLVMPass.cpp
  SpecializeLinalgMatmulPass.cpp
  HexmemCpyToDMA.cpp
  FusionPass.cpp
  SplitReduceGenericPass.cpp
  EraseVectorToTensorWritebackPass.cpp
  SlicingPass.cpp
  TilingPass.cpp
  VectorizationPass.cpp
  RewriteUBPoisonToZeroPass.cpp
  VTCMTiling.cpp
  VTCMMaskWidening.cpp
  LowerPack.cpp
  VTCMTilingOptions.cpp
  LowerConstantsSeparately.cpp
  SplitReductionUtils.cpp
  ConversionToFp16Pass.cpp
  OptimizeExtfTruncfOpPass.cpp
  HexagonVectorizePackedGenericPass.cpp
  LLVMAlignmentPropagationPass.cpp
  SetAliasScopesPass.cpp
  PromoteTiledSCFLoopToVTCMPass.cpp
  LinalgToCompositePass.cpp
  LowerCompositePass.cpp
  ThreadDistribution.cpp

  DEPENDS
  LinalgToLLVMConversionPassIncGen
  HexagonBackendTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRComposite
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRSupport
  MLIRRegisterAllExtensions
  MLIRPresburger
  MLIRAnalysis

  Transforms
)
