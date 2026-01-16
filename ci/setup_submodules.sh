# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "Configuring git for submodules"
cd "${REPO_ROOT}"
# Add submodules if missing
if [ ! -d "triton" ]; then
  git submodule add --force https://github.com/triton-lang/triton.git triton
  cd triton
  echo "Applying qcom specific patches to triton"
  git checkout e44bd1c83c1c3e8deac7c4f02683cfb3cc395c8b
  git apply "${REPO_ROOT}/third_party_software/patches/triton/third_party_triton.patch"
fi

cd "${REPO_ROOT}"
if [ ! -d "triton_shared" ]; then
  git submodule add --force https://github.com/microsoft/triton-shared triton_shared
  cd triton_shared
  git checkout 2b728ad97bc02af821a0805b09075838911d4c19
  echo "Applying qcom specific patches to triton_shared"
  git apply "${REPO_ROOT}/third_party_software/patches/triton_shared/max_with_nan_propagation.patch"
  git apply "${REPO_ROOT}/third_party_software/patches/triton_shared/tt_shared_split_dim.patch"
fi

echo "Submodules triton and triton_shared initialized and patched successfully."
