# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_conversion_library(ShardToLLVM
  ShardToLLVMPass.cpp

  DEPENDS
    ShardToLLVMConversionPassIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRShardDialect
    MLIRTensorDialect
    MLIRTransforms
    MLIRSupport
    HexagonCommon
)
