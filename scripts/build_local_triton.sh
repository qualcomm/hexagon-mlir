#!/usr/bin/env bash 
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail
set -x

ROOT="$(git rev-parse --show-toplevel)"
CHECK_SCRIPT="$ROOT/scripts/check_local_env.sh"

echo "=== Running environment checks ==="
source "$CHECK_SCRIPT"

echo "=== Activating Python environment ==="
# CONDA_ENV is validated by check_local_env.sh
source "$CONDA_ENV/bin/activate"

echo "=== Upgrading pip tooling ==="
pip install --upgrade pip setuptools wheel

echo "=== Building Triton ==="
cd "$ROOT/triton"

TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=true \
LLVM_INCLUDE_DIRS="$LLVM_PROJECT_BUILD_DIR/include" \
LLVM_LIBRARY_DIR="$LLVM_PROJECT_BUILD_DIR/lib" \
LLVM_SYSPATH="$LLVM_PROJECT_BUILD_DIR" \
pip install -e . --no-build-isolation --verbose

echo "🎉 Triton build completed successfully."

cd "$ROOT"
set +x
