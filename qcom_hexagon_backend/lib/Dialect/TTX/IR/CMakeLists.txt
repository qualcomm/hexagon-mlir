# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

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
