# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

if (NOT DEFINED ENV{HEXAGON_SDK_ROOT})
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not set.  Make sure that the Hexagon SDK has been set up.")
endif()
if (NOT DEFINED ENV{HEXAGON_ARCH_VERSION})
    message(FATAL_ERROR "HEXAGON_ARCH_VERSION not set.  Please set the Hexagon arch version according to target device e.g., 75.")
endif()
add_subdirectory(multithreading)

set(HEXAGON_MLIR_ROOT_LOCAL $ENV{HEXAGON_MLIR_ROOT})
set(LLVM_BIN_DIR $ENV{LLVM_SYSPATH}/bin)
message(STATUS "LLVM BIN DIR is ${LLVM_BIN_DIR}")

###############################################################################
# TODO: Enable using the cmake file from the SDK
# include($ENV{HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake)
# This fails now as we invoke both x86 compiler and the hexagon compiler in this
# file.
# For now pick up relevant variables from the HexagonSDK
###############################################################################


set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
set(HEXKL_ROOT $ENV{HEXKL_ROOT})
set(Q6_VERSION $ENV{HEXAGON_ARCH_VERSION})

# Get the Binary extension of the Hexagon Toolchain
if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
    set(HEXAGON_TOOLCHAIN_SUFFIX .exe)
endif()

if (DEFINED ENV{HEXAGON_TOOLS})
    set(HEXAGON_TOOLS $ENV{HEXAGON_TOOLS})
else()
    message(FATAL_ERROR "HEXAGON_TOOLS not set.")
endif()

# Accept either a toolchain root containing bin/ and target/, or an archive
# directory containing Tools/. Normalize to the directory that directly
# contains bin/, target/, and lib/.
if (NOT EXISTS "${HEXAGON_TOOLS}/bin/hexagon-clang++" AND
    EXISTS "${HEXAGON_TOOLS}/Tools/bin/hexagon-clang++")
    set(HEXAGON_TOOLS "${HEXAGON_TOOLS}/Tools")
endif()

message ("Using Hexagon_Tools: ${HEXAGON_TOOLS}")

set(HEXAGON_C_COMPILER ${HEXAGON_TOOLS}/bin/hexagon-clang${HEXAGON_TOOLCHAIN_SUFFIX})
set(HEXAGON_CXX_COMPILER ${HEXAGON_TOOLS}/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})
set(HEXAGON_AR ${HEXAGON_TOOLS}/bin/hexagon-ar${HEXAGON_TOOLCHAIN_SUFFIX})


set(HEXAGON_INCLUDES
    -I${HEXAGON_SDK_ROOT}/incs
    -I${HEXAGON_SDK_ROOT}/incs/stddef
    -I${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc
    -I${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc/internal
    -I${HEXKL_ROOT}/include
    -isystem ${HEXAGON_SDK_ROOT}/rtos/qurt/computev${Q6_VERSION}/include/posix
    -isystem ${HEXAGON_SDK_ROOT}/rtos/qurt/computev${Q6_VERSION}/include/qurt
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    # -I${HEXAGON_TOOLS}/target/hexagon/include/c++/v1
)


# Cap bitcode arch at v75: at -mv79, the QHL header (qhmath_hvx_vector.h) includes
# xqf/ which uses IEEE FP intrinsics (vmpy.sf.sf.128B) that hit an upstream LLVM
# ISel/MC-verifier bug. Capping at v75 selects the QFloat implementation instead.
if(Q6_VERSION GREATER 75)
  set(BITCODE_ARCH 75)
else()
  set(BITCODE_ARCH ${Q6_VERSION})
endif()
set(HEXAGON_FLAGS -O2 -Wall -mv${BITCODE_ARCH} -mhvx -mhvx-qfloat -mhvx-ieee-fp)

###############################################################################
# Build a library of runtime linked modules
###############################################################################
set(HEXAGON_RUNTIME_SRC
    IntrinsicsHVX.c
    IntrinsicsF8E4M3Fstar.c
    UserDMA/UserDMA.cc
    UserDMA/RuntimeDMA.cc
    src/VTCMPool.cpp
    src/HexagonBuffer.cpp
    src/HexagonBufferAlias.cpp
    src/HexagonAPI.cpp
    src/HexagonCAPI.cpp
    src/PrintRuntimeDSP.cc
)

add_library(hexagon_runtime STATIC)

if(HEXKL_ROOT)
    message(STATUS "HEXKL_ROOT set; enabling HexKL runtime sources")
    list(APPEND HEXAGON_INCLUDES -I${HEXKL_ROOT}/include)
    list(APPEND HEXAGON_RUNTIME_SRC
        HMXMicroDaemon/HMXMicroDaemon.cc
        src/HexKLAPI.cpp
    )
    target_compile_definitions(hexagon_runtime INTERFACE
        HEXAGON_MLIR_ENABLE_HEXKL_RUNTIME=1
    )
else()
    message(STATUS "HEXKL_ROOT not set; excluding HexKL runtime sources")
endif()

foreach (i IN LISTS HEXAGON_RUNTIME_SRC)
    get_filename_component(src_name ${i} NAME_WE)
    get_filename_component(src_ext ${i} EXT)
    if(src_ext STREQUAL ".cpp" OR src_ext STREQUAL ".cc")
        set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${i})
        set(COMPILER ${HEXAGON_CXX_COMPILER})
    elseif(src_ext STREQUAL ".c")
        set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${i})
        set(COMPILER ${HEXAGON_C_COMPILER})
    else()
        message(FATAL_ERROR "Cannot find source file: ${i}.{c/cpp/cc}")
    endif()
    set(BC ${src_name}.bc)
    set(BC_2_CPP ${src_name}_bc.cpp)
    get_filename_component(BASENAME ${src_name} NAME)
    # We need to make sure that CPLUS_INCLUDE_PATH is empty for successful compilation on Ubuntu 22 (as for Ubuntu 22 it contains the path of libstdc++'s headers files *for x86*)
    add_custom_command(OUTPUT ${BC}
                     COMMAND env CPLUS_INCLUDE_PATH="" ${COMPILER} ${HEXAGON_INCLUDES} ${HEXAGON_FLAGS} -emit-llvm -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -c ${SRC_FILE} -o ${BC}
                     DEPENDS ${SRC_FILE}
                     VERBATIM)
    add_custom_command(OUTPUT ${BC_2_CPP}
                       COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/bitcode2array.py ${BC} ${BASENAME} ${BC_2_CPP}
                       DEPENDS ${BC} ${CMAKE_CURRENT_SOURCE_DIR}/bitcode2array.py
                       VERBATIM)
    target_sources(hexagon_runtime PRIVATE ${BC_2_CPP})
endforeach()

###############################################################################
# Build tests as an external project to use the hexagon toolchain compiler
###############################################################################

# Lists don't pass through External project calls, so convert them to string to
# be passed to the external project
string(REPLACE ";" " " HEXAGON_FLAGS_STR "${HEXAGON_FLAGS}")
string(REPLACE ";" " " HEXAGON_INCLUDES_STR "${HEXAGON_INCLUDES}")
#Unit tests test implementation, they require implementation headers in addition
#to interface headers
set(RUNTIME_UNIT_TEST_INCLUDES_STR -I${CMAKE_CURRENT_SOURCE_DIR}/UserDMA)
string(REPLACE ";" " " HEXAGON_RUNTIME_SRC_STR "${HEXAGON_RUNTIME_SRC}")

include(ExternalProject)

set(CMAKE_ARGS_LIST
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}"
  "-DCMAKE_CXX_COMPILER=${HEXAGON_CXX_COMPILER}"
  "-DCMAKE_C_COMPILER=${HEXAGON_C_COMPILER}"
)

list(APPEND CMAKE_ARGS_LIST "-DCMAKE_CXX_FLAGS=${HEXAGON_FLAGS_STR} ${HEXAGON_INCLUDES_STR} ${RUNTIME_UNIT_TEST_INCLUDES_STR}"
"-DCMAKE_C_FLAGS=${HEXAGON_FLAGS_STR} ${HEXAGON_INCLUDES_STR} ${RUNTIME_UNIT_TEST_INCLUDES_STR}"
"-DCMAKE_ASM_FLAGS=${HEXAGON_FLAGS_STR} ${HEXAGON_INCLUDES_STR}")

list(APPEND CMAKE_ARGS_LIST
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DCMAKE_CXX_STANDARD=17"
  "-DHEXAGON_RUNTIME_SRC_STR=${HEXAGON_RUNTIME_SRC_STR}"
  "-DBASE_BINARY_DIR=${CMAKE_BINARY_DIR}"
  "-DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}"
  "-DHEXAGON_TOOLS=${HEXAGON_TOOLS}"
  "-DHEXAGON_C_COMPILER=${HEXAGON_C_COMPILER}"
  "-DHEXAGON_CXX_COMPILER=${HEXAGON_CXX_COMPILER}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)

ExternalProject_Add(
    hexagon_runtime_test_project
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test
    CMAKE_ARGS ${CMAKE_ARGS_LIST}
    BUILD_ALWAYS 1
)
