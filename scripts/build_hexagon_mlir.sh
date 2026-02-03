#!/bin/bash
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail
set -e

echo "----------------------------------------------------"
echo "Welcome to hexagon-mlir local building script ..    "
echo "----------------------------------------------------"

REPO_DIR="$(git rev-parse --show-toplevel)"
echo "REPO_DIR=${REPO_DIR}"
BASE_DIR="$(cd .. && pwd)"
echo "BASE_DIR=${BASE_DIR}"

# get triton and triton-shared
cd ${REPO_DIR}
# Check if triton or triton_shared directories already exist
if [[ ! -d "${REPO_DIR}/triton" ]]; then
    echo "Initializing submodules..."
    source ci/setup_submodules.sh
else
    echo "Submodules already initialized. Skipping."
fi

# Get HOST_TOOLCHAIN
cd ${BASE_DIR}
echo "extracting HOST_TOOLCHAIN..."
mkdir -p HOST_TOOLCHAIN
cd HOST_TOOLCHAIN  

if [[ ! -f clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz ]]; then 
  echo "Downloading HOST_TOOLCHAIN..." 
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
fi

if [[ ! -d bin ]]; then
  echo "Extracting HOST_TOOLCHAIN..."
  tar -xf clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz --strip-components=1
else  
  echo "HOST_TOOLCHAIN already extracted. Skipping."
fi

export HOST_TOOLCHAIN=${BASE_DIR}/HOST_TOOLCHAIN
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"

# Get HEXAGON_SDK
cd ${BASE_DIR}
echo "extracting HEXAGON_SDK..."
mkdir -p HEXAGON_SDK
cd HEXAGON_SDK/

if [[ ! -f Hexagon_SDK_lnx.zip ]]; then
  echo "Downloading HEXAGON_SDK..."
  wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip
fi

if [[ ! -d Hexagon_SDK ]]; then
  echo "Extracting HEXAGON_SDK..."
  unzip Hexagon_SDK_lnx.zip
else
  echo "HEXAGON_SDK already extracted. Skipping."
fi
export HEXAGON_SDK_ROOT=${BASE_DIR}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/

# Get HEXAGON_TOOLS
cd ${BASE_DIR}
echo "extracting HEXAGON_TOOLS..."
mkdir -p HEXAGON_TOOLS
cd HEXAGON_TOOLS

if [[ ! -f Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz ]]; then
  echo "Downloading HEXAGON_TOOLS..."
  wget https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_open_access/Linux/Debian/19.0.02/Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
fi
if [[ ! -d Tools ]]; then
  echo "Extracting HEXAGON_TOOLS..."
  tar -xzf Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
else
  echo "HEXAGON_TOOLS already extracted. Skipping."
fi
HEXAGON_TOOLS=${BASE_DIR}/HEXAGON_TOOLS/Tools
export HEXAGON_TOOLS=${HEXAGON_TOOLS}

# Get Hexagon_KL
echo "extracting Hexagon_KL..."
cd ${BASE_DIR}
mkdir -p HEXKL_DIR
cd HEXKL_DIR

if [[ ! -f Hexagon_KL.Core.1.0.0.Linux-Any.zip ]]; then
  echo "Downloading Hexagon_KL..."
  wget https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/Hexagon_KL.Core.1.0.0.Linux-Any.zip
fi

if [[ ! -d hexkl_addon ]]; then
  echo "Extracting Hexagon_KL..."
  unzip -q Hexagon_KL.Core.1.0.0.Linux-Any.zip
  unzip -q hexkl-1.0.0-beta1-6.4.0.0.zip
else
  echo "Hexagon_KL already extracted. Skipping."
fi
export HEXKL_ROOT=${BASE_DIR}/HEXKL_DIR/hexkl_addon

# check BASE_DIR is pre-defined in the environment
: "${BASE_DIR:?Please set BASE_DIR before running this script}"

# Derived paths
echo "Clone and build LLVM..."
LLVM_TOP_DIR="${BASE_DIR}/LLVM_DIR"
LLVM_SRC_DIR="${LLVM_TOP_DIR}/llvm-project"
LLVM_PROJECT_BUILD_DIR="${LLVM_SRC_DIR}/build"
LLVM_INSTALL_DIR="${LLVM_PROJECT_BUILD_DIR}/install"

cd ${BASE_DIR}
mkdir -p "${LLVM_TOP_DIR}"
cd ${LLVM_TOP_DIR}

# Clone llvm-project if it doesn't exist
if [[ ! -d "${LLVM_SRC_DIR}/.git" ]]; then
  echo "Cloning llvm-project..."
  git clone https://github.com/llvm/llvm-project.git "${LLVM_SRC_DIR}"
else
  echo "llvm-project already exists. Skipping clone."
fi
cd ${LLVM_SRC_DIR}

# Pin to a specific commit for reproducibility
git checkout 064f02dac0c81c19350a74415b3245f42fed09dc


if [[ ! -f "${LLVM_PROJECT_BUILD_DIR}/bin/mlir-opt" ]]; then
  mkdir -p ${LLVM_PROJECT_BUILD_DIR}
  cd  ${LLVM_PROJECT_BUILD_DIR} 
  echo "Configuring CMake..."
  cmake \
    -G "Unix Makefiles" \
    -S "${LLVM_SRC_DIR}/llvm" \
    -B "${LLVM_PROJECT_BUILD_DIR}" \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;Hexagon" \
    -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_CCACHE_BUILD:BOOL=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_BUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_DEFAULT_TARGET_TRIPLE="x86_64-unknown-linux-gnu" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"

  echo "Compiling and installing LLVM..."
  cmake --build "${LLVM_PROJECT_BUILD_DIR}" -j
  cmake --install "${LLVM_PROJECT_BUILD_DIR}"
else
  echo "LLVM already built. Skipping build."
fi

export LLVM_PROJECT_BUILD_DIR=${LLVM_PROJECT_BUILD_DIR}

# setup environment for test and execution.
ENV_DIR="${BASE_DIR}/mlir-env"
cd ${BASE_DIR}
if [[ ! -d "$ENV_DIR" ]]; then
  echo "Creating virtual environment in '$ENV_DIR'..."
  python3 -m venv "$ENV_DIR"
else
  echo "Virtual environment directory '$ENV_DIR' already exists; skipping creation."
fi
# Activate it
source "$ENV_DIR/bin/activate"


export CONDA_ENV=${BASE_DIR}/mlir-env

cd ${REPO_DIR}
source scripts/build_triton.sh

if [[ -d ./triton/build ]]; then
    lit ./triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/test/
else
    echo "Triton build missing; skipping tests."
fi
echo "Local build script completed successfully."
