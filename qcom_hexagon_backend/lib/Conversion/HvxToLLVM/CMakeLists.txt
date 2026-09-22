add_mlir_conversion_library(HvxToLLVM
  HvxToLLVMPass.cpp

  DEPENDS
  HvxToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  Hvx
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
