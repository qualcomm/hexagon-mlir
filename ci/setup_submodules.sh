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

# Ensure existing submodules are initialized
git submodule update --init

add_and_checkout() {
	  local name="$1"
	    local url="$2"
	      local commit="$3"

  cd "${REPO_ROOT}"
  if ! git submodule status "${name}" &>/dev/null; then
	      echo "Adding submodule ${name}"
	          git submodule add "${url}" "${name}"
		    fi

  echo "Checking out ${name} at ${commit}"
    cd "${REPO_ROOT}/${name}"
      git fetch origin
        git checkout "${commit}"
}

add_and_checkout \
	  triton \
	    https://github.com/triton-lang/triton.git \
	      df38505e451a1541555379bcf378be9e8c00545c
	      
add_and_checkout \
	  triton_shared \
	    https://github.com/facebookincubator/triton-shared.git \
	      0614763d270ec0eacba9d5d8283cdff6bedb03c8
	      
cd "${REPO_ROOT}"
echo "Applying qcom specific patches to triton_shared"
bash "${REPO_ROOT}/ci/apply_patches.sh" || {
	  echo "ERROR: Failed while applying patches"
	    exit 1
    }
    
echo "Submodules triton and triton_shared initialized and patched successfully."
