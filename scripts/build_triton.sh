#!/usr/bin/env bash 
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail
set -x

HEXAGON_MLIR_ROOT="$(git rev-parse --show-toplevel)"
export HEXAGON_MLIR_ROOT=$HEXAGON_MLIR_ROOT

echo "=== Upgrading pip tooling ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing triton and torch-mlir build requirements ==="
pip install -r ${HEXAGON_MLIR_ROOT}/ci/requirements.txt

echo "=== Building triton ==="
cd "$HEXAGON_MLIR_ROOT/triton"

export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton

# Get the Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Triton shared path
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

export HEXAGON_ARCH_VERSION=75
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:${PYTHONPATH:-}

TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=true \
LLVM_INCLUDE_DIRS="$LLVM_PROJECT_BUILD_DIR/include" \
LLVM_LIBRARY_DIR="$LLVM_PROJECT_BUILD_DIR/lib" \
LLVM_SYSPATH="$LLVM_PROJECT_BUILD_DIR" \
pip install -e . --no-build-isolation --verbose

echo "🎉 Triton build completed successfully."
cd "$HEXAGON_MLIR_ROOT"

set +x
