# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LINALG_HEXAGON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LINALG_HEXAGON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

add_subdirectory(include)

# Umbrella ordering target for every TableGen-generated (.inc) header in the
# backend. Many library targets include a dialect/conversion header from a
# *sibling* subdirectory that pulls in a generated .inc file; LINK_LIBS does
# not create the compile-order edge those sources need, so under -j they can
# race ahead of TableGen ("'...Dialect.h.inc' file not found"). Any target that
# includes a generated header should list HexagonBackendTableGen in its DEPENDS
# (in addition to its own IncGen target). When a new dialect/conversion is
# added, register its TableGen target here once and every consumer is covered.
# Declared after add_subdirectory(include) so all IncGen targets already exist,
# and before add_subdirectory(lib) so lib targets can depend on it.
add_custom_target(HexagonBackendTableGen)
add_dependencies(HexagonBackendTableGen
  AffineToLLVMConversionPassIncGen
  CroutonTableGen
  DMAToLLVMConversionPassIncGen
  HexKLTableGen
  HexKLToLLVMConversionPassIncGen
  HexagonMemTableGen
  HexagonMemToLLVMConversionPassIncGen
  HexagonTPtrTableGen
  LinalgToLLVMConversionPassIncGen
  ShardToLLVMConversionPassIncGen
  TTXTableGen
  TmTensorTableGen
  TransformsPassIncGen
  TritonTableGen
)

add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_triton_plugin(TritonHexagon
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Triton/RewriteI64PtrArgsToI32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Triton/EliminateSafeBoundaryChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Triton/RewriteMaskedAccessToInboundsFastPath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/LinkRuntimeModules.cpp

    DEPENDS
    TritonTableGen
    HexagonBackendTableGen

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
