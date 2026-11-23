# ===- test_dsp_bitonic_sort.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

import triton

from .. import dsp

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def bitonic_step_kernel(
    data_ptr,
    idx_ptr,
    n: tl.constexpr,  # number of elements per row (power of 2)
    j: tl.constexpr,  # current compare distance (j = k>>1, ... , 1)
    k: tl.constexpr,  # current sequence size (k = 2, 4, 8, ..., n)
    descending: tl.constexpr,  # False => ascending, True => descending
    BLOCK_SIZE: tl.constexpr,  # should be >= n; typically set to n
):
    row_id = tl.program_id(0)
    row_start = row_id * n

    i = tl.arange(0, BLOCK_SIZE)
    in_bounds = i < n

    # Load row values + indices
    vals = tl.load(data_ptr + row_start + i, mask=in_bounds, other=0)
    ids = tl.load(idx_ptr + row_start + i, mask=in_bounds, other=0)

    # Compute partner indices within the row: ixj = i ^ j
    ixj = i ^ j
    # Gather partner vals/ids from memory. (You can also keep in registers across steps
    # if you fuse loops, but this simple form is clearer.)
    p_vals = tl.load(data_ptr + row_start + ixj, mask=in_bounds, other=0)
    p_ids = tl.load(idx_ptr + row_start + ixj, mask=in_bounds, other=0)

    # Bitonic direction for this (k, j) step:
    # Lower half (i & k == 0) sorts ascending; upper half sorts descending.
    # Apply global 'descending' by XOR'ing.
    ascend_here = ((i & k) == 0) ^ descending

    # Stable comparison: smaller first if ascending; larger first if descending.
    # When vals equal, tie-break on original indices (ids).
    is_less = (vals < p_vals) | ((vals == p_vals) & (ids < p_ids))
    is_greater = (vals > p_vals) | ((vals == p_vals) & (ids > p_ids))

    # For each position i:
    #  - If we're in ascending part, lane with (i & j)==0 takes min; lane with (i & j)!=0 takes max.
    #  - If descending, the opposite.
    left_lane = (i & j) == 0
    right_lane = ~left_lane

    # Decide if this lane should take the "smaller" or "larger" of the pair.
    take_small = tl.where(ascend_here, left_lane, right_lane)

    # Select output values/ids for this lane
    out_vals = tl.where(
        take_small, tl.where(is_less, vals, p_vals), tl.where(is_greater, vals, p_vals)
    )
    # For equal & equal-ids cases (shouldn't happen), keep current 'vals'/'ids'.
    # The expressions above already handle tie-break by ids.

    # Now set matching ids accordingly
    out_ids = tl.where(
        take_small, tl.where(is_less, ids, p_ids), tl.where(is_greater, ids, p_ids)
    )

    # Store back
    tl.store(data_ptr + row_start + i, out_vals, mask=in_bounds)
    tl.store(idx_ptr + row_start + i, out_ids, mask=in_bounds)


def bitonic_sort_triton(x: torch.Tensor, dim: int = -1, descending: bool = False):
    # Move sort dim to last
    x = x.movedim(dim, -1).contiguous()
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, "Input size along 'dim' must be a power of two"

    # Flatten all but last dim to rows
    x_flat = x.reshape(-1, n).contiguous()
    n_rows = x_flat.shape[0]

    # Working buffers
    data = x_flat.clone()
    idx = (
        torch.arange(n, device=x.device, dtype=torch.int32)
        .expand(n_rows, n)
        .contiguous()
    )

    # Flatten for device kernels
    data_1d = data.reshape(-1)
    idx_1d = idx.reshape(-1)

    # One program per row; vector width = n
    grid = (n_rows,)
    BLOCK_SIZE = n

    # Canonical bitonic network
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            bitonic_step_kernel[grid](
                data_1d,
                idx_1d,
                n,
                j,
                k,
                descending,
                BLOCK_SIZE,
            )
            j >>= 1
        k <<= 1

    # Reshape back
    data = data_1d.reshape(n_rows, n).reshape(orig_shape)
    idx = idx_1d.reshape(n_rows, n).reshape(orig_shape)

    # Move dimension back to original
    data = data.movedim(-1, dim)
    idx = idx.movedim(-1, dim)
    return data, idx


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_bitonic_sort():
    x = torch.tensor([[3, 7, 4, 8, 6, 2, 1, 5]], dtype=torch.int32)

    output, output_idx = bitonic_sort_triton(x, dim=-1, descending=False)
    reference, reference_idx = torch.sort(x, dim=-1, stable=True)

    assert torch.equal(output, reference)
    assert torch.equal(output_idx, reference_idx)
