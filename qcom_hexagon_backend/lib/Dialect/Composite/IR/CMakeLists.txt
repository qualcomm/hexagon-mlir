# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_dialect_library(MLIRComposite
  CompositeDialect.cpp
  CompositeOps.cpp

  DEPENDS
  CompositeTableGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRTensorDialect
  MLIRTilingInterface
)
