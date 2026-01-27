add_mlir_conversion_library(AffineToLLVM
  AffineToLLVM.cpp
  AffineToLLVMPass.cpp
  AffineTilingPass.cpp
  AffinePipelineFusionPass.cpp
  AffineVectorize.cpp
  AffineTileMemoryPass.cpp
  Utils/BareissLU.cpp
  Utils/MatroidIntersection.cpp

  DEPENDS
  AffineToLLVMConversionPassIncGen

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
  MLIRSupport
)
