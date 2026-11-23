# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_conversion_library(AffineToLLVM
  AffineToLLVM.cpp
  AffineToLLVMPass.cpp
  AffineTilingPass.cpp
  AffinePipelineFusionPass.cpp
  AffineVectorize.cpp
  AffineTileMemoryPass.cpp
  Utils/BareissLU.cpp
  Utils/MatroidIntersection.cpp

  DEPENDS
  AffineToLLVMConversionPassIncGen
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
)
