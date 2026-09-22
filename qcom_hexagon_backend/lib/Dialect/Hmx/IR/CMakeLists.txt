add_mlir_dialect_library(Hmx
  HmxOps.cpp
  HmxDialect.cpp
  HmxAttrs.cpp

  DEPENDS
  HmxTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
)
