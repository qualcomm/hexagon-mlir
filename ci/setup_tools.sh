#!/usr/bin/env bash 
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -euo pipefail

# Hexagon SDK Version and URL 
SDK_VERSION="6.4.0.2"
SDK_ZIP="Hexagon_SDK_lnx.zip"
SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${SDK_VERSION}/${SDK_ZIP}"

# Install directory
INSTALL_DIR="/local/mnt/workspace/MLIR_build_artifacts"
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

# HexKL 
KL_VERSION="1.0.0"
KL_OUTER_ZIP="Hexagon_KL.Core.${KL_VERSION}.Linux-Any.zip"
KL_URL="https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/${KL_VERSION}/${KL_OUTER_ZIP}"

KL_BASE="${INSTALL_DIR}/Hexagon_KL"
KL_DIR="${KL_BASE}/${KL_VERSION}"

mkdir -p "${KL_BASE}"
echo "Downloading Hexagon KL version ${KL_VERSION}..."

wget -q --show-progress "${KL_URL}" -O "${KL_BASE}/${KL_OUTER_ZIP}"

INNER_ZIP="hexkl-1.0.0-beta1-6.4.0.0.zip"
# Make sure inner zip exists inside the outer zip
unzip -q -j "${KL_BASE}/${KL_OUTER_ZIP}" "${INNER_ZIP}" -d "${KL_BASE}"
echo "Extracting Hexagon KL..."
unzip -q "${KL_BASE}/${INNER_ZIP}" -d "${KL_DIR}"

# Locate hexkl_addon directory 
HEXKL_ADDON_DIR=$(find "${KL_DIR}" -type d -name "hexkl_addon" | head -n 1)
if [ -z "${HEXKL_ADDON_DIR}" ]; then
    echo "Error: hexkl_addon directory not found in ${KL_DIR}"
    exit 1
fi

export HEXKL_ROOT="${HEXKL_ADDON_DIR}"

echo "HEXKL_ROOT=${HEXKL_ROOT}"

echo "Hexagon SDK, Tools, and HexKL have been set up successfully."
exit 0
