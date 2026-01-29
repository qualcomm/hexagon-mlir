#!/usr/bin/env bash 
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -x
export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
export LLVM_INSTALL_DIR=$LLVM_PROJECT_BUILD_DIR 

source ${HEXAGON_MLIR_ROOT}/ci/setup_triton_env.sh

pip install --upgrade pip setuptools wheel

# Build Triton using upstream LLVM
cd $TRITON_ROOT
echo Building triton

TRITON_BUILD_WITH_CLANG_LLD=1 \
    TRITON_BUILD_WITH_CCACHE=true \
    LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include \
    LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib \
    LLVM_SYSPATH=$LLVM_INSTALL_DIR \
    pip install  -e . --no-build-isolation --verbose
if [ $? -ne 0 ]; then
    echo Building hexagon-mlir failed
else
    echo hexagon-mlir successfully built
fi

cd $HEXAGON_MLIR_ROOT

set +x
