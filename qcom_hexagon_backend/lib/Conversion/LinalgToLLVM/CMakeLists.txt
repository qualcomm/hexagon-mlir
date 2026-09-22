add_mlir_conversion_library(LinalgToLLVM
  Common.cpp
  ExtendPackUnpackFrontier.cpp
  FormSCFThreads.cpp
  FormVirtualThreadsPass.cpp
  FormAsyncThreadsPass.cpp
  LinalgToLLVM.cpp
  LinalgToLLVMPass.cpp
  HexmemCpyToDMA.cpp
  FusionPass.cpp
  SplitReduceGenericPass.cpp
  VectorRowReducePass.cpp
  EraseVectorToTensorWritebackPass.cpp
  SlicingPass.cpp
  TilingPass.cpp
  VectorizationPass.cpp
  RewriteUBPoisonToZeroPass.cpp
  VTCMTiling.cpp
  LowerPack.cpp
  VTCMTilingOptions.cpp
  LowerConstantsSeparately.cpp
  SplitReductionUtils.cpp
  ConversionToFp16Pass.cpp
  OptimizeExtfTruncfOpPass.cpp
  
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
  # LinalgToLLVMPass.cpp calls into the HMX dialect transforms (attribution,
  # tile level, weight residency) and these conversion passes
  HmxTransforms
  HexagonMemToLLVM
  HmxToLLVM
  HexKLToLLVM
  DMAToLLVM
  # the vector-row-reduce pass emits hvx.vror; the pipeline lowers it via
  # HvxToLLVM
  Hvx
  HvxToLLVM
)
