# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
#!/usr/bin/env bash 
set -x

export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$PWD/triton

# Get the Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Triton shared path
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

export HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
export HEXAGON_TOOLS=${HEXAGON_TOOLS}
export HEXKL_ROOT=${HEXKL_ROOT}
export HEXAGON_ARCH_VERSION=75
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:$PYTHONPATH

# Add host toolchain to PATH
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"

set +x
