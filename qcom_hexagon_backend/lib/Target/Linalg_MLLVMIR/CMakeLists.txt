# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_translation_library(TritonLinalgLLVM
        MLLVMIRTranslation.cpp

        DEPENDS
        HexagonBackendTableGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTargetLLVMIRExport
        )
