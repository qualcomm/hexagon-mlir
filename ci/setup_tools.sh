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

