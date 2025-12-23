add_mlir_dialect_library(TTX
  TTXAttrs.cpp
  TTXTypes.cpp
  TTXOps.cpp
  TTXDialect.cpp

  DEPENDS
  TTXTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
)
