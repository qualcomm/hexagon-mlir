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
