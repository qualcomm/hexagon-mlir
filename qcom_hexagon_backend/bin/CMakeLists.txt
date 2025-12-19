add_subdirectory(runtime)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

###############################################################################
# hexagon-translate
###############################################################################

add_llvm_executable(linalg-hexagon-translate
    linalg-hexagon-translate.cpp
    LinkRuntimeModules.cpp
    PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(linalg-hexagon-translate)
target_link_libraries(linalg-hexagon-translate PRIVATE
    HexagonMLIRLLVMToLLVMIR
    ${dialect_libs}
    ${conversion_libs}
    # tests
    TritonTestAnalysis

    LLVMCore
    LLVMSupport
    LLVMOption
    LLVMCodeGen
    LLVMAsmParser

    # MLIR core
    MLIROptLib
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRExecutionEngine
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRTransformUtils
    MLIRLLVMToLLVMIRTranslation

    # Hexagon runtime routines
    hexagon_runtime
)

mlir_check_all_link_libraries(linalg-hexagon-translate)

###############################################################################
# linalg-hexagon-opt
###############################################################################

add_llvm_executable(linalg-hexagon-opt linalg-hexagon-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(linalg-hexagon-opt)
target_link_libraries(linalg-hexagon-opt PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    # tests
    TritonTestAnalysis

    LLVMCore
    LLVMSupport
    LLVMOption
    LLVMCodeGen
    LLVMAsmParser

    # MLIR core
    MLIROptLib
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRExecutionEngine
    MLIRMathToLLVM
    MLIRTransformUtils
    MLIRLLVMToLLVMIRTranslation
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
)

mlir_check_all_link_libraries(linalg-hexagon-opt)

###############################################################################
# hexagon-mlir-lsp-server
###############################################################################

add_llvm_executable(hexagon-mlir-lsp-server hexagon-mlir-lsp-server.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(hexagon-mlir-lsp-server)
target_link_libraries(hexagon-mlir-lsp-server PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    # tests
    TritonTestAnalysis

    # MLIR core
    MLIRIR
    MLIRLspServerLib
    MLIRRegisterAllExtensions
    MLIRRegisterAllDialects
)

mlir_check_all_link_libraries(hexagon-mlir-lsp-server)
