# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

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

