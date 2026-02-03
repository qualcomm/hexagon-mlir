#!/bin/bash
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail
set -x

export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$PWD/triton
BASE_DIR="$(cd .. && pwd)"

# Get the Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Triton shared path
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

echo "BASE_DIR=${BASE_DIR}"
HEXAGON_TOOLS=${BASE_DIR}/HEXAGON_TOOLS/Tools
export HEXAGON_TOOLS=${HEXAGON_TOOLS}
export HEXAGON_SDK_VERSION=6.4.0.2
export HEXAGON_SDK_ROOT=${BASE_DIR}/HEXAGON_SDK/Hexagon_SDK/$HEXAGON_SDK_VERSION
export HEXKL_ROOT=${BASE_DIR}/HEXKL_DIR/hexkl_addon

export HEXAGON_ARCH_VERSION=75
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PATH=/prj/qct/llvm/devops/build_tools/triton/adb_tool:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:$PYTHONPATH
