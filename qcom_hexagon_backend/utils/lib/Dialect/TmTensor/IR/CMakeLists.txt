add_mlir_dialect_library(TmTensor
  TmTensorAttrs.cpp
  TmTensorTypes.cpp
  TmTensorOps.cpp
  TmTensorDialect.cpp

  DEPENDS
  TmTensorTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
)
