#!/usr/bin/env bash 
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail

echo "=== Hexagon MLIR Local Environment Check ==="

ROOT="$(git rev-parse --show-toplevel)"

# Required environment variables
REQUIRED_VARS=(
  "HEXAGON_SDK_ROOT"
  "HEXAGON_TOOLS"
  "HEXKL_ROOT"
  "LLVM_PROJECT_BUILD_DIR"
  "CONDA_ENV"
)

echo ""
echo "Checking required environment variables..."

for VAR in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR:-}" ]; then
    echo "❌ Missing: $VAR"
    MISSING=1
  else
    echo "✔ $VAR = ${!VAR}"
  fi
done

if [ "${MISSING:-0}" -eq 1 ]; then
  echo ""
  echo "Some required variables are missing. Fix them and re-run this script."
  exit 1
fi

echo ""
echo "Checking directory existence..."

check_dir() {
  local NAME="$1"
  local PATH="$2"

  if [ ! -d "$PATH" ]; then
    echo "❌ $NAME directory not found at: $PATH"
    exit 1
  else
    echo "✔ $NAME directory exists"
  fi
}

check_dir "HEXAGON_SDK_ROOT" "$HEXAGON_SDK_ROOT"
check_dir "HEXAGON_TOOLS" "$HEXAGON_TOOLS"
check_dir "HEXKL_ROOT" "$HEXKL_ROOT"
check_dir "LLVM_PROJECT_BUILD_DIR" "$LLVM_PROJECT_BUILD_DIR"
check_dir "CONDA_ENV" "$CONDA_ENV"

echo ""
echo "Checking LLVM build..."

if [ ! -f "$LLVM_PROJECT_BUILD_DIR/bin/mlir-opt" ]; then
  echo "❌ LLVM build incomplete: mlir-opt not found in:"
  echo "   $LLVM_PROJECT_BUILD_DIR/bin"
  exit 1
else
  echo "✔ LLVM build contains mlir-opt"
fi

echo ""
echo "Checking Triton submodule..."

if [ ! -d "$ROOT/triton" ]; then
  echo "❌ Triton submodule missing. Run:"
  echo "   git submodule update --init --recursive"
  exit 1
else
  echo "✔ Triton submodule present"
fi

echo ""
echo "Checking Python environment..."

if [ ! -f "$CONDA_ENV/bin/python" ]; then
  echo "❌ Python interpreter not found in CONDA_ENV"
  exit 1
fi

PYVER="$("$CONDA_ENV/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "✔ Python version in CONDA_ENV: $PYVER"

echo ""
echo "🎉 Environment looks good. You are ready to build Triton."
