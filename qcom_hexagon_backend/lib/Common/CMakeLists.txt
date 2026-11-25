# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_library(HexagonCommon STATIC
     Common.cpp
     DMATransferUtil.cpp
)
target_link_libraries(HexagonCommon PUBLIC MLIRMemRefUtils)
