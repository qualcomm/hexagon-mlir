add_mlir_dialect_library(HmxTransforms
  BufferizableOpInterfaceImpl.cpp
  MatmulToHmxPass.cpp
  HmxPartitionPass.cpp
  WeightResidentPass.cpp
  HmxWorkspaceResidentPass.cpp

  DEPENDS
  HmxTransformsPassIncGen

  LINK_LIBS PUBLIC
  HexagonCommon
  HexagonMem
  Hmx
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
