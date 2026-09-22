add_mlir_translation_library(TritonLinalgLLVM
        MLLVMIRTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTargetLLVMIRExport
        # MLLVMIRTranslation.cpp calls hexagon::createLinalgToLLVMPass()
        LinalgToLLVM
        )
