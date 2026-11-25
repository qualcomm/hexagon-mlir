# ===- test_dsp_stable_sort.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""
Stable bitonic sort implementation for Hexagon backend.

This test implements a stable bitonic sort algorithm using Triton. The implementation
uses a row-based grid pattern where each program processes one complete row.

Key implementation details:
- XOR-based partner indexing on pure tl.arange() for Hexagon backend compatibility
- Row-based grid: one program per row with row_start = row_id * n (n is constexpr)
- Stable sorting via tie-breaking on original indices
- Input padded to power-of-2 for bitonic sort requirements
"""

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
def bitonic_merge_kernel(
    data_ptr,
    indices_ptr,
    n: tl.constexpr,
    stage: tl.constexpr,
    step: tl.constexpr,
    descending: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bitonic merge kernel for stable sorting. One program processes one row."""
    row_id = tl.program_id(0)
    row_start = row_id * n

    i = tl.arange(0, BLOCK_SIZE)
    in_bounds = i < n

    data = tl.load(data_ptr + row_start + i, mask=in_bounds, other=float("inf"))
    indices = tl.load(indices_ptr + row_start + i, mask=in_bounds, other=n)

    # Compute pair indices using XOR on pure arange result
    pair_distance = 1 << (step - 1)
    ixj = i ^ pair_distance
    pair_data = tl.load(data_ptr + row_start + ixj, mask=in_bounds, other=float("inf"))
    pair_indices = tl.load(indices_ptr + row_start + ixj, mask=in_bounds, other=n)

    # Determine bitonic direction for this stage
    k = 1 << (stage + 1)
    ascend_here = ((i & k) == 0) ^ descending

    # Stable comparison with tie-breaking on indices
    is_less = (data < pair_data) | ((data == pair_data) & (indices < pair_indices))
    is_greater = (data > pair_data) | ((data == pair_data) & (indices > pair_indices))

    left_lane = (i & pair_distance) == 0
    right_lane = ~left_lane
    take_small = tl.where(ascend_here, left_lane, right_lane)
    new_data = tl.where(
        take_small,
        tl.where(is_less, data, pair_data),
        tl.where(is_greater, data, pair_data),
    )
    new_indices = tl.where(
        take_small,
        tl.where(is_less, indices, pair_indices),
        tl.where(is_greater, indices, pair_indices),
    )
    tl.store(data_ptr + row_start + i, new_data, mask=in_bounds)
    tl.store(indices_ptr + row_start + i, new_indices, mask=in_bounds)


def stable_sort_triton(x: torch.Tensor, dim: int = -1, descending: bool = False):
    """Stable sort using Triton bitonic sort algorithm."""
    x = x.movedim(dim, -1)
    orig_shape = x.shape
    n_elements = orig_shape[-1]
    x_flat = x.reshape(-1, n_elements).contiguous()
    n_rows = x_flat.shape[0]
    sorted_data = x_flat.clone()
    indices = torch.arange(n_elements, device=x.device, dtype=torch.int32)
    indices = indices.unsqueeze(0).expand(n_rows, -1).contiguous()
    n_padded = 2 ** ((n_elements - 1).bit_length())

    if n_padded > n_elements:
        pad_size = n_padded - n_elements
        sorted_data = torch.cat(
            [
                sorted_data,
                torch.full(
                    (n_rows, pad_size), float("inf"), device=x.device, dtype=x.dtype
                ),
            ],
            dim=1,
        )
        pad_indices = torch.arange(
            n_elements, n_padded, device=x.device, dtype=torch.int32
        )
        indices = torch.cat(
            [indices, pad_indices.unsqueeze(0).expand(n_rows, -1)], dim=1
        )

    BLOCK_SIZE = n_padded
    sorted_data = sorted_data.reshape(-1)
    indices = indices.reshape(-1)
    grid = (n_rows,)
    num_stages = (n_padded - 1).bit_length()
    for stage in range(num_stages):
        for step in range(stage + 1, 0, -1):
            bitonic_merge_kernel[grid](
                sorted_data,
                indices,
                n_padded,  # Pass row size as constexpr
                stage,
                step,
                descending,
                BLOCK_SIZE,
            )

    sorted_data = sorted_data.reshape(n_rows, n_padded)[:, :n_elements]
    indices = indices.reshape(n_rows, n_padded)[:, :n_elements]
    sorted_data = sorted_data.reshape(orig_shape)
    indices = indices.reshape(orig_shape)
    sorted_data = sorted_data.movedim(-1, dim)
    indices = indices.movedim(-1, dim)
    return sorted_data, indices


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_stable_sort():
    x = torch.randn(4, 8)

    output, output_idx = stable_sort_triton(x, dim=-1, descending=False)
    reference, reference_idx = torch.sort(x, dim=-1, stable=True)

    assert torch.allclose(output, reference, rtol=1e-5)


@dsp
def test_stable_sort_dup():
    x = torch.tensor([[3.0, 1.0, 2.0, 1.0, 3.0]])

    output, output_idx = stable_sort_triton(x, dim=-1)
    reference, reference_idx = torch.sort(x, dim=-1, stable=True)

    assert torch.equal(output_idx, reference_idx)
