#!/bin/bash
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

set -euxo pipefail

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
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:${PYTHONPATH:-}

# libtriton.so requires GLIBCXX_3.4.30. Miniconda's Python binary has an
# RPATH of $ORIGIN/../lib which causes its older libstdc++ to be resolved
# before the system version, regardless of LD_LIBRARY_PATH. LD_PRELOAD
# overrides RPATH. Use HOST_TOOLCHAIN's clang (the same compiler that built
# libtriton.so) to locate the libstdc++ it linked against, without hardcoding
# any system path.
_HOST_TOOLCHAIN=${BASE_DIR}/HOST_TOOLCHAIN
_LIBSTDCPP=$("${_HOST_TOOLCHAIN}/bin/clang" --print-file-name=libstdc++.so.6 2>/dev/null || true)
if [[ -n "${_LIBSTDCPP}" && -f "${_LIBSTDCPP}" ]]; then
    export LD_PRELOAD="${_LIBSTDCPP}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
unset _HOST_TOOLCHAIN _LIBSTDCPP

set +euxo pipefail
