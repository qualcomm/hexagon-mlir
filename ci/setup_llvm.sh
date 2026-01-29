#!/usr/bin/env bash
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Setup HOST_TOOLCHAIN environment variable
export HOST_TOOLCHAIN="/local/mnt/workspace/MLIR_build_artifacts/host_toolchain/"
# if HOST_TOOLCHAIN does not exist, download and set it up
if [ ! -d "${HOST_TOOLCHAIN}" ]; then
  echo "HOST_TOOLCHAIN not found at ${HOST_TOOLCHAIN}. Downloading and setting up..."
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
  mkdir -p "${HOST_TOOLCHAIN}"
  tar -xf clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz -C "${HOST_TOOLCHAIN}" --strip-components=1
fi
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"

# Verify clang installation paths exist
if [ ! -f "${HOST_TOOLCHAIN}/bin/clang" ] || [ ! -f "${HOST_TOOLCHAIN}/bin/clang++" ]; then
  echo "Error: clang or clang++ not found in HOST_TOOLCHAIN at ${HOST_TOOLCHAIN}/bin/"
  exit 1
fi
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"

# Read the expected LLVM commit hash for Triton
# Check if the file exists
if [ ! -f "${REPO_ROOT}/triton/cmake/llvm-hash.txt" ]; then
  echo "Error: Expected LLVM hash file not found at ${REPO_ROOT}/triton/cmake/llvm-hash.txt"
  exit 1
fi
EXPECTED_LLVM_HASH=$(cat "${REPO_ROOT}/triton/cmake/llvm-hash.txt" | tr -d '[:space:]')
echo "Expected LLVM commit hash for Triton: ${EXPECTED_LLVM_HASH}"

# Setup LLVM installation directory
LLVM_INSTALL_DIR="/local/mnt/workspace/MLIR_build_artifacts/llvm_triton"
mkdir -p "${LLVM_INSTALL_DIR}"

# Clone the LLVM repository if not already present
if [ ! -d "${LLVM_INSTALL_DIR}/llvm-project" ]; then
  echo "Cloning LLVM repository..."
  git clone -b main https://github.com/llvm/llvm-project.git "${LLVM_INSTALL_DIR}/llvm-project"
fi

cd "${LLVM_INSTALL_DIR}/llvm-project"
echo "Checking out LLVM commit ${EXPECTED_LLVM_HASH}..."
git fetch origin
git checkout "${EXPECTED_LLVM_HASH}"
echo "Checked out LLVM commit:"
git rev-parse HEAD

# Build LLVM with required projects
BUILD_DIR="${LLVM_INSTALL_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring LLVM build with CMake..."
cmake -G "Ninja" ../llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld" \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_ASM_COMPILER=${CC} \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;Hexagon" \
    -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_CCACHE_BUILD:BOOL=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_BUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

echo "Building LLVM..."
ninja -j$(nproc)

echo "Installing LLVM..."
ninja install

echo "LLVM installed at ${BUILD_DIR}/install"
echo "Host toolchain located at ${HOST_TOOLCHAIN}"
