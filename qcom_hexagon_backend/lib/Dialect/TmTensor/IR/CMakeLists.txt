# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_dialect_library(TmTensor
  TmTensorAttrs.cpp
  TmTensorTypes.cpp
  TmTensorOps.cpp
  TmTensorDialect.cpp

  DEPENDS
  TmTensorTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRDestinationStyleOpInterface
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
)
