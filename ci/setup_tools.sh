# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
#!/usr/bin/env bash 
set -euo pipefail

# Hexagon SDK Version and URL 
SDK_VERSION="6.4.0.2"
SDK_ZIP="Hexagon_SDK_lnx.zip"
SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${SDK_VERSION}/${SDK_ZIP}"

# Install directory
INSTALL_DIR="$HOME/hexagon-mlir-artifacts"
SDK_DIR="${INSTALL_DIR}/Hexagon_SDK/${SDK_VERSION}"

mkdir -p "${INSTALL_DIR}"

echo "Downloading Hexagon SDK version ${SDK_VERSION}..."
wget -q --show-progress "${SDK_URL}" -O "${INSTALL_DIR}/${SDK_ZIP}"

echo "Extracting Hexagon SDK..."
unzip -q "${INSTALL_DIR}/${SDK_ZIP}" -d "${INSTALL_DIR}"

echo "Hexagon SDK installed at ${SDK_DIR}"
export HEXAGON_SDK_ROOT="${SDK_DIR}"

echo "Setting HEXAGON_SDK_ROOT to ${HEXAGON_SDK_ROOT}"

# Source the environment setup script
source "${HEXAGON_SDK_ROOT}/setup_sdk_env.source"

# Hexagon Toolchain 
TOOLCHAIN_VERSION="19.0.02"
TOOLCHAIN_TAR="Hexagon_open_access.Core.${TOOLCHAIN_VERSION}.Linux-Any.tar.gz"
TOOLCHAIN_URL="https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_open_access/Linux/Debian/${TOOLCHAIN_VERSION}/${TOOLCHAIN_TAR}"

echo "Downloading Hexagon Toolchain version ${TOOLCHAIN_VERSION}..."
wget -q --show-progress "${TOOLCHAIN_URL}" -O "${INSTALL_DIR}/${TOOLCHAIN_TAR}"

echo "Extracting Hexagon Toolchain..."
tar -xzf "${INSTALL_DIR}/${TOOLCHAIN_TAR}" -C "${INSTALL_DIR}"

export HEXAGON_TOOLS="${INSTALL_DIR}/Tools"
echo "Setting HEXAGON_TOOLS to ${HEXAGON_TOOLS}"