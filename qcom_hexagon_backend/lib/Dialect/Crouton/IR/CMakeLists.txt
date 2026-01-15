add_mlir_dialect_library(Crouton
  CroutonTypes.cpp
  CroutonDialect.cpp

  DEPENDS
  CroutonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
)
