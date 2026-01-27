add_mlir_dialect_library(HexagonMem
  HexagonMemAttrs.cpp
  HexagonMemTypes.cpp
  HexagonMemOps.cpp
  HexagonMemDialect.cpp

  DEPENDS
  HexagonMemTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
  HexagonCommon
)
