list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LINALG_HEXAGON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LINALG_HEXAGON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_triton_plugin(TritonHexagon
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/LinkRuntimeModules.cpp

    LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    TritonLinalgLLVM
    AffineToLLVM
    LinalgToLLVM
    DMAToLLVM
    HexKLToLLVM
    HexagonMemToLLVM
    HexagonMLIRLLVMToLLVMIR
    Transforms

    MLIRRegisterAllDialects

    # Hexagon runtime routines
    hexagon_runtime
)
