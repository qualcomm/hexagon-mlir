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
# Triton shared patch: default the arith-to-linalg pass to transposeReduceToRank0=false so
# reductions are lowered without the legacy transpose. The reduction-lowering patch below relies
# on this default.
TRITON_SHARED_REMOVE_TRANSPOSED_REDUCTION_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_remove_tranposed_reduction.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_REMOVE_TRANSPOSED_REDUCTION_PATCH_FILE"

# Triton shared patch for multi-result tt.reduce lowering (fused sum/sum-sq, etc). Rewrites
# convertToLinalgReduce and relies on the transposeReduceToRank0=false default set above, so it
# must apply after the remove-transposed patch.
TRITON_SHARED_MULTI_RESULT_REDUCTION_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton_shared/triton_shared_multi_result_reduction.patch"
apply_patch_if_needed "$HEXAGON_MLIR_ROOT/triton_shared" "$TRITON_SHARED_MULTI_RESULT_REDUCTION_PATCH_FILE"

# -----------------------------------------------------------------------------
# Triton patches
# -----------------------------------------------------------------------------
# Add libdevice sigmoid support to triton
TRITON_LIBDEVICE_SIGMOID_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/libdevice_sigmoid.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_LIBDEVICE_SIGMOID_PATCH_FILE"

TRITON_ALIGNMENT_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/pass_alignment_info_to_kernel.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_ALIGNMENT_PATCH_FILE"

# Triton patch adding an in-bounds fast-path for tensor-descriptor accesses.
# RewriteTensorDescriptorToPointer gains an `inbounds-fast-path` option that
# guards the lowered load/store with a scalar in-bounds check, so fully
# in-bounds tiles take an unmasked fast path and only boundary tiles pay for
# mask computation. Also bundles the env-gated frontend TMA lowering
# (TRITON_HEXAGON_TMA_INBOUNDS_FAST_PATH) in semantic.py as a default-off
# alternative.
TRITON_REWRITE_DESCRIPTOR_INBOUNDS_FAST_PATH_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/rewrite_descriptor_inbounds_fast_path.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_REWRITE_DESCRIPTOR_INBOUNDS_FAST_PATH_PATCH_FILE"

TRITON_WHL_BUILD_PATCH_FILE="$HEXAGON_MLIR_ROOT/third_party_software/patches/triton/enable_wheel_build.patch"
apply_patch_if_needed "$TRITON_ROOT" "$TRITON_WHL_BUILD_PATCH_FILE"
