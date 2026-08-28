#!/usr/bin/env bash
# 
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt


set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "Configuring git submodules"
cd "${REPO_ROOT}"


add_and_checkout() {
	  local name="$1"
	    local url="$2"
	      local commit="$3"

  cd "${REPO_ROOT}"
  if [ ! -d "${REPO_ROOT}/${name}" ]; then
	      echo "Cloning ${name}"
	          git clone "${url}" "${name}"
		    fi

  echo "Checking out ${name} at ${commit}"
    cd "${REPO_ROOT}/${name}"
      git fetch origin
        git checkout "${commit}"
}

add_and_checkout \
	  triton \
	    https://github.com/triton-lang/triton.git \
	      a9ced836206b1495b31d9519d86077b5152572ea
	      
add_and_checkout \
	  triton_shared \
	    https://github.com/facebookincubator/triton-shared.git \
	      f2eb8c5eda10f3c8aec11c72d4d4d0f6346ec89d
	      
cd "${REPO_ROOT}"
echo "Applying qcom specific patches to triton_shared"
bash "${REPO_ROOT}/ci/apply_patches.sh" || {
	  echo "ERROR: Failed while applying patches"
	    exit 1
    }
    
echo "Submodules triton and triton_shared initialized and patched successfully."
