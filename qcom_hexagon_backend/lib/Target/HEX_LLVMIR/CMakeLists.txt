# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

add_mlir_translation_library(HexagonMLIRLLVMToLLVMIR
    LLVMIRTranslation.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRTargetLLVMIRExport

    # LLVM libraries
    LLVMHexagonAsmParser
    LLVMHexagonCodeGen
    LLVMHexagonInfo
    LLVMHexagonDesc
    LLVMHexagonDisassembler
)
