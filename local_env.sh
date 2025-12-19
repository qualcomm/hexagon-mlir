#!/bin/bash
set -x
export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$PWD/triton

# Get the Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Triton shared path
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

#Triton Driver env setup
# Temporary fix for 21.0 nightly toolset failures
export HEXAGON_TOOLS=/prj/qct/llvm/release/internal/HEXAGON/branch-21.0/linux64/toolset-2304/Tools
export HEXAGON_ARCH_VERSION=75
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PATH=/prj/qct/llvm/devops/build_tools/triton/adb_tool:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:$PYTHONPATH

# Override the Hexagon SDK Root path. 
# (Note): Modify locally if using a local HEXAGON_SDK and/or needs to make edits to the SDK.
export HEXAGON_SDK_VERSION=6.4.0.0
export HEXAGON_SDK_ROOT=/prj/qct/llvm/devops/build_tools/triton/Hexagon_SDK/$HEXAGON_SDK_VERSION
export HEXKL_ROOT=/prj/qct/llvm/devops/build_tools/triton/HexKL/HEXKL_REL_2025_09_21_1_0_0_BETA/sdkl_addon

set +x
