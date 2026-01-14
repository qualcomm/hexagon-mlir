add_mlir_dialect_library(HexKL
  HexKLOps.cpp
  HexKLDialect.cpp

  DEPENDS
  HexKLTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
)
