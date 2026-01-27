add_mlir_conversion_library(LinalgToLLVM
  Common.cpp
  FormSCFThreads.cpp
  FormVirtualThreadsPass.cpp
  FormAsyncThreadsPass.cpp
  LinalgToLLVM.cpp
  LinalgToLLVMPass.cpp
  HexmemCpyToDMA.cpp
  FusionPass.cpp
  SlicingPass.cpp
  TilingPass.cpp
  VectorizationPass.cpp
  RewriteUBPoisonToZeroPass.cpp
  VTCMTiling.cpp
  VTCMTilingOptions.cpp
  LowerConstantsSeparately.cpp
  SplitReductionUtils.cpp

  DEPENDS
  LinalgToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRSupport
  MLIRRegisterAllExtensions

  Transforms
)
