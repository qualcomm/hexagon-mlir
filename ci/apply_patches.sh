#!/bin/bash
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
set -Eeuox pipefail

# If the CI image accidentally has /etc/gitconfig as a *directory*, Git will fail
# when reading the system config. We detect that here and warn that this script
# will ignore the system gitconfig by using GIT_CONFIG_NOSYSTEM=1 on all git calls.
if [ -d /etc/gitconfig ]; then
  echo "WARNING: Detected /etc/gitconfig is a DIRECTORY; system gitconfig will be ignored via GIT_CONFIG_NOSYSTEM=1 for all git operations in this script."
fi

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
HEXAGON_MLIR_ROOT="$(readlink -f "$SCRIPT_DIR/../")"
TRITON_ROOT="$HEXAGON_MLIR_ROOT/triton"

# Apply a patch if it isn't already applied (stateless; no marker file).
apply_patch_if_needed() {
    local repo_dir="$1"   # e.g., "$TRITON_ROOT" or "$HEXAGON_MLIR_ROOT/triton_shared"
    local patch_file="$2" # e.g., ".../patches/triton/third_party_triton.patch"

    if [ ! -f "$patch_file" ]; then
        echo "WARNING: Patch file not found at $patch_file"
        return 0
    fi

    echo "Checking/applying patch: $patch_file in $repo_dir"
    pushd "$repo_dir" >/dev/null

    # 1) If the reverse applies, the patch is already present — skip.
    if GIT_CONFIG_NOSYSTEM=1 git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
        echo "Patch already applied (reverse-check passed): $patch_file — skipping."
        popd >/dev/null
        return 0
    fi

    # 2) Try to apply the patch directly.
    #    git apply is the source of truth to avoid TOCTOU races.
    if GIT_CONFIG_NOSYSTEM=1 git apply "$patch_file"; then
        echo "Patch applied successfully: $patch_file"
        popd >/dev/null
        return 0
    fi

    # 3) Neither forward nor reverse apply cleanly -> inconsistent state / conflicts.
    echo "ERROR: Patch neither applies nor is already applied: $patch_file"
    echo "----- git apply --check (verbose) output -----"
    GIT_CONFIG_NOSYSTEM=1 git apply --check -v "$patch_file" || true
    echo "----------------------------------------------"
    popd >/dev/null
    exit 1
}

# -----------------------------------------------------------------------------
# Apply patches (drop the marker basename arg; keep the order if patches depend on one another)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# triton_shared patches
# -----------------------------------------------------------------------------
# Triton shared patch to update the API for compatibility with the latest LLVM
TRITON_SHARED_API_UPDATE_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_3_6_triton.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_API_UPDATE_PATCH_FILE"

# Triton shared patch on Pointer Analysis
TRITON_SHARED_POINTER_ANALYSIS_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_ptr_analysis.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_POINTER_ANALYSIS_PATCH_FILE"

# Triton shared patch on split pointers
TRITON_SHARED_SPLIT_DIM_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_split_dim.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_SPLIT_DIM_PATCH_FILE"

# Triton shared patch to handle canonicalization pattern of Max with NaN propagation
TRITON_SHARED_MAX_NAN_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_max_nan.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_MAX_NAN_PATCH_FILE"

# -----------------------------------------------------------------------------
# Triton patches (build third-party backends + NVVM ReductionKind compatibility)
# -----------------------------------------------------------------------------
# Triton patch to get around NVVM ReductionKind compatibility issue
TRITON_NVVM_COMPATIBILITY_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/nvvm_reduction_kind_compatibility.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_NVVM_COMPATIBILITY_PATCH_FILE"

# Add libdevice sigmoid support to triton
TRITON_LIBDEVICE_SIGMOID_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/libdevice_sigmoid.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_LIBDEVICE_SIGMOID_PATCH_FILE"
