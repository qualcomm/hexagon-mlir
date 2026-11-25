# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_conversion_library(HexagonMemToLLVM
  HexagonMemToLLVMPass.cpp
  HexagonMemExternalFnNames.cpp

  DEPENDS
  HexagonMemToLLVMConversionPassIncGen
  HexagonBackendTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  HexagonCommon
)
