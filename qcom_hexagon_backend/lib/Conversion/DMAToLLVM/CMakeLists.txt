# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_conversion_library(DMAToLLVM
  DMAExternalFnNames.cpp
  DMAToLLVMPass.cpp

  DEPENDS
  DMAToLLVMConversionPassIncGen
  HexagonBackendTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  HexagonCommon
)

