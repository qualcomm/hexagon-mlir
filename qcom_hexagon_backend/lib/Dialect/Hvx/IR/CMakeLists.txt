add_mlir_dialect_library(Hvx
  HvxOps.cpp
  HvxDialect.cpp

  DEPENDS
  HvxTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
