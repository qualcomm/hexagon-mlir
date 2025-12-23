add_mlir_conversion_library(DMAToLLVM
  DMAExternalFnNames.cpp
  DMAToLLVMPass.cpp

  DEPENDS
  DMAToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  HexagonCommon
)

