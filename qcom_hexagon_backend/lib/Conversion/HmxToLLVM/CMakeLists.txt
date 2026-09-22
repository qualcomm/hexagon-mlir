add_mlir_conversion_library(HmxToLLVM
  HmxToLLVMPass.cpp
  HmxExternalFnNames.cpp

  DEPENDS
  HmxToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  Hmx
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
