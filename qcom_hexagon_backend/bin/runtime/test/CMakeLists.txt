cmake_minimum_required(VERSION 3.18)
project(hexagon_runtime_test)

set(SOURCE_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../")

set(HEXAGON_RUNTIME_TEST_SRCS
    test/hexagon_runtime_test.cpp
    test/VTCMPoolTests.cpp
    test/HexagonBufferTests.cpp
    test/HexagonAPITests.cpp
    test/HexagonCAPITests.cpp
    test/UserDMATests.cpp
    test/UserDMA2DTests.cpp
    test/HexagonBufferAliasTests.cpp
)

# Convert the files back to list
string(REPLACE " " ";" HEXAGON_RUNTIME_SRC "${HEXAGON_RUNTIME_SRC_STR}")

# Temp workaround to ignore this file since it contains undefined symbols and
# causes error in run_main_on_hexagon

list(APPEND HEXAGON_RUNTIME_TEST_SRCS "${HEXAGON_RUNTIME_SRC}")

set(HEXAGON_RUNTIME_TEST_SRCS_WITH_PREFIX)

foreach(ELEMENT IN LISTS HEXAGON_RUNTIME_TEST_SRCS)
    list(APPEND HEXAGON_RUNTIME_TEST_SRCS_WITH_PREFIX "${SOURCE_PREFIX}${ELEMENT}")
endforeach()

set(SHARED_SOURCES ${HEXAGON_RUNTIME_TEST_SRCS_WITH_PREFIX})

add_library(hexagon_runtime_test SHARED ${SHARED_SOURCES})

set(PREBUILT_LIB_DIR "${HEXAGON_TOOLCHAIN}/Tools/lib")
set(USE_HEXAGON_GTEST "${HEXAGON_SDK_ROOT}/utils/googletest/gtest")
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(FetchContent)
FetchContent_Declare(googletest SOURCE_DIR "${USE_HEXAGON_GTEST}")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
target_link_libraries(hexagon_runtime_test PUBLIC gtest)
target_include_directories(hexagon_runtime_test PRIVATE "${USE_HEXAGON_GTEST}/include")
set_target_properties(hexagon_runtime_test PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")

install(TARGETS hexagon_runtime_test DESTINATION lib)

set(TEST_PATH "${CMAKE_BINARY_DIR}/test/")
set(BENCHMARK "libhexagon_runtime_test")
set(DEVICE_HOST "$ENV{ANDROID_HOST}")
set(DEVICE_SERIAL "$ENV{ANDROID_SERIAL}")
set(DEVICE_PATH "/vendor/bin/${BENCHMARK}")
set(Q6_VERSION "$ENV{HEXAGON_ARCH_VERSION}")
set(DSP_STACK_SIZE "262144")

# Generate the run_test.sh script to run the test
# TODO: Add option to pass gtest args in commandline
set(SHELL_SCRIPT_CONTENT "#!/bin/bash"
  "set -x"
  "adb -H \$ANDROID_HOST -s \$ANDROID_SERIAL push ${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon ${DEVICE_PATH}/run_main_on_hexagon"
  "adb -H \$ANDROID_HOST -s \$ANDROID_SERIAL push ${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/hexagon_toolv87_v${Q6_VERSION}/librun_main_on_hexagon_skel.so /vendor/lib/rfsa/adsp/"
  "adb -H \$ANDROID_HOST -s \$ANDROID_SERIAL push ${TEST_PATH}/${BENCHMARK}.so ${DEVICE_PATH}"
  "adb -H \$ANDROID_HOST -s \$ANDROID_SERIAL shell 'cd ${DEVICE_PATH}\; touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf\; ./run_main_on_hexagon 3 ${DEVICE_PATH}/${BENCHMARK}.so --gtest-args=\"\" stack_size=262144'"
  "adb -H \$ANDROID_HOST -s \$ANDROID_SERIAL pull ${DEVICE_PATH}/gtest_output.txt ."
  "cat gtest_output.txt"
)

list(JOIN SHELL_SCRIPT_CONTENT "\n" SHELL_SCRIPT_CONTENT_STR)

file(WRITE ${CMAKE_BINARY_DIR}/run_test.sh "${SHELL_SCRIPT_CONTENT_STR}")

# Make the shell script executable
file(CHMOD ${CMAKE_BINARY_DIR}/run_test.sh FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Copy to base binary directory (something like python/build/cmake.linux-x86_64-cpython-3.11)
file(COPY ${CMAKE_BINARY_DIR}/run_test.sh DESTINATION ${BASE_BINARY_DIR})

# Add a custom target to run the shell script
add_custom_target(run_test
    COMMAND ${CMAKE_BINARY_DIR}/run_test.sh
    DEPENDS ${CMAKE_BINARY_DIR}/run_test.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the test script"
)
