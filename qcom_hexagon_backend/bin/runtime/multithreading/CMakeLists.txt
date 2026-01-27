project(MLIRAsyncRuntime)
if (NOT DEFINED ENV{HEXAGON_SDK_ROOT})
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not set.  Make sure that the Hexagon SDK has been set up.")
endif()

set(HEXAGON_MLIR_ROOT_LOCAL $ENV{HEXAGON_MLIR_ROOT})

# Triton driver expects HEXAGON_TOOLS instead of HEXAGON_TOOLS_ROOT
if (DEFINED ENV{HEXAGON_TOOLS})
    set(HEXAGON_TOOLS_ROOT $ENV{HEXAGON_TOOLS}/..)
elseif (DEFINED ENV{HEXAGON_TOOLS_ROOT})
    set(HEXAGON_TOOLS_ROOT $ENV{HEXAGON_TOOLS_ROOT})
else()
    message(FATAL_ERROR "HEXAGON_TOOLS not set.")
endif()

message ("Using Hexagon_Tools: ${HEXAGON_TOOLS_ROOT}")

set(HEXAGON_TOOLCHAIN ${HEXAGON_TOOLS_ROOT})
set(Q6_VERSION $ENV{HEXAGON_ARCH_VERSION})

# Set the compilers to hexagon-clang and hexagon-clang++
set(CMAKE_C_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_CXX_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})

message ("MULTITHREADING: cmake cxx compiler : ${CMAKE_CXX_COMPILER}")


message(STATUS "Building MLIR AsyncRuntime package") 


set(LLVM_ROOT_DIR $ENV{LLVM_SYSPATH})

get_filename_component(LLVM_ROOT ${LLVM_ROOT_DIR} DIRECTORY)

set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})

if(NOT DEFINED LLVM_ROOT)
    message(FATAL_ERROR "LLVM_ROOT not set.")
endif()

# Include directories for QURT libraries
include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/rtos/qurt/computev${Q6_VERSION}/include/posix
    ${HEXAGON_SDK_ROOT}/rtos/qurt/computev${Q6_VERSION}/include/qurt
    ${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/include/c++/v1/__bit
)

# Make a directory for fake includes to avoid issues with missing system headers
message(STATUS "Creating fake include directory at ${CMAKE_BINARY_DIR}/fake_includes")
set (FAKE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/fake_includes)
file(MAKE_DIRECTORY ${FAKE_INCLUDE_DIR}/machine)
file(WRITE ${FAKE_INCLUDE_DIR}/machine/endian.h 
"#pragma once\n #include <endian.h>\n")

include_directories(BEFORE ${FAKE_INCLUDE_DIR})
message(STATUS "Fake include directory added to include path")

# Include directories for LLVM
include_directories(
    ${LLVM_ROOT_DIR}/include
)

# Compiler flags


# Source files
set(SOURCES
    AsyncRuntime.cpp
    HexagonThreadPool.cpp
    HexagonCustomDefs.cpp
    StringMap.cpp
    MemAlloc.cpp
)

# Compiler flags
add_definitions(
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -fPIC
    -Werror=date-time
    -Werror=unguarded-availability-new
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wwrite-strings
    -Wcast-qual
    -Wmissing-field-initializers
    -pedantic
    -Wno-long-long
    -Wimplicit-fallthrough
    -Wcovered-switch-default
    -Wno-noexcept-type
    -Wnon-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Wno-comment
    -Wstring-conversion
    -fdiagnostics-color
    -ffunction-sections
    -fdata-sections
    -O3
    -DNDEBUG
    -std=c++17
    #-fno-exceptions
    -funwind-tables
    -fno-rtti
)

if(TARGET hexagon_mlir_async_runtime)
  message(STATUS "Target 'hexagon_mlir_async_runtime' already exists.")
  get_target_property(_type hexagon_mlir_async_runtime TYPE)
  message(STATUS "Target type: ${_type}")
endif()

# Create the shared library
add_library(hexagon_mlir_async_runtime STATIC ${SOURCES})

# Remove manual linkage of stdc++fs 
get_target_property(existing_libs hexagon_mlir_async_runtime LINK_LIBRARIES)
list(REMOVE_ITEM existing_libs stdc++fs)
set_target_properties(hexagon_mlir_async_runtime PROPERTIES LINK_LIBRARIES "${existing_libs}")

# Set the library's name
set_target_properties(hexagon_mlir_async_runtime PROPERTIES OUTPUT_NAME "hexagon_mlir_async_runtime")
