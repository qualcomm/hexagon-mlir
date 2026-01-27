add_mlir_conversion_library(HexKLToLLVM
  HexKLToLLVMPass.cpp
  HexKLExternalFnNames.cpp

  DEPENDS
  HexKLToLLVMConversionPassIncGen

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
