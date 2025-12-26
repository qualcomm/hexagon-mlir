add_mlir_dialect_library(TPtr
  HexagonTPtrOps.cpp
  HexagonTPtrDialect.cpp

  DEPENDS
  HexagonTPtrTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPtrDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
)

