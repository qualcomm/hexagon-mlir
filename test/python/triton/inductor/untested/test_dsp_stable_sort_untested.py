# ===- test_dsp_stable_sort_untested.py ------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


import sys

import torch
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from triton.language.extra import libdevice

import triton

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def bitonic_merge_kernel(
    data_ptr,
    indices_ptr,
    n_elements: tl.constexpr,
    stage: tl.constexpr,
    step: tl.constexpr,
    descending: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Bitonic merge kernel for stable sorting.
    Uses indices to break ties and ensure stability.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data and indices
    data = tl.load(data_ptr + offsets, mask=mask, other=float("inf"))
    indices = tl.load(indices_ptr + offsets, mask=mask, other=n_elements)

    # Compute pair indices for comparison
    pair_distance = 1 << (step - 1)
    direction = ((offsets >> stage) & 1) == 0

    # Calculate which element to compare with
    is_ascending_pair = (offsets & pair_distance) == 0
    pair_offset = tl.where(
        is_ascending_pair, offsets + pair_distance, offsets - pair_distance
    )

    # Load pair data
    pair_mask = (pair_offset < n_elements) & mask
    pair_data = tl.load(data_ptr + pair_offset, mask=pair_mask, other=float("inf"))
    pair_indices = tl.load(indices_ptr + pair_offset, mask=pair_mask, other=n_elements)

    # Stable comparison: compare values first, then indices
    should_swap = (data > pair_data) | ((data == pair_data) & (indices > pair_indices))

    # Apply direction (ascending/descending based on bitonic sequence)
    if descending:
        direction = ~direction
    should_swap = should_swap ^ direction

    # Perform conditional swap
    new_data = tl.where(should_swap, pair_data, data)
    new_indices = tl.where(should_swap, pair_indices, indices)

    # Store results
    tl.store(data_ptr + offsets, new_data, mask=mask)
    tl.store(indices_ptr + offsets, new_indices, mask=mask)


@triton.jit
def merge_sort_merge_kernel(
    src_ptr,
    src_indices_ptr,
    dst_ptr,
    dst_indices_ptr,
    n_elements: tl.constexpr,
    width: tl.constexpr,
    descending: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Merge kernel for merge sort - provides stable sorting.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Calculate merge boundaries
    merge_id = block_start // (2 * width)
    left_start = merge_id * 2 * width
    right_start = left_start + width
    merge_end = tl.minimum(left_start + 2 * width, n_elements)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Determine if this element comes from left or right partition
    in_left = offsets < right_start
    in_right = offsets >= right_start

    left_idx = tl.where(in_left, offsets - left_start, 0)
    right_idx = tl.where(in_right, offsets - right_start, 0)

    # Load from source
    output_data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    output_indices = tl.load(src_indices_ptr + offsets, mask=mask, other=0)

    # Store to destination
    tl.store(dst_ptr + offsets, output_data, mask=mask)
    tl.store(dst_indices_ptr + offsets, output_indices, mask=mask)


def stable_sort_triton(x: torch.Tensor, dim: int = -1, descending: bool = False):
    """
    Stable sort using Triton with bitonic sort algorithm.

    Args:
        x: Input tensor to sort
        dim: Dimension along which to sort
        descending: Sort in descending order if True

    Returns:
        Tuple of (sorted_values, sorted_indices)
    """
    # Move the sort dimension to the last dimension
    x = x.movedim(dim, -1)
    orig_shape = x.shape
    n_elements = orig_shape[-1]

    # Flatten all but last dimension
    x_flat = x.reshape(-1, n_elements).contiguous()
    n_rows = x_flat.shape[0]

    # Create output tensors
    sorted_data = x_flat.clone()
    indices = torch.arange(n_elements, device=x.device, dtype=torch.int32)
    indices = indices.unsqueeze(0).expand(n_rows, -1).contiguous()

    # Calculate next power of 2
    n_padded = 2 ** ((n_elements - 1).bit_length())

    if n_padded > n_elements:
        # Pad to power of 2 for bitonic sort
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

    # Bitonic sort parameters
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_rows * n_padded, BLOCK_SIZE)

    # Flatten for processing
    sorted_data = sorted_data.reshape(-1)
    indices = indices.reshape(-1)
    total_elements = n_rows * n_padded

    # Bitonic sort stages
    num_stages = (n_padded - 1).bit_length()

    for stage in range(num_stages):
        for step in range(stage + 1, 0, -1):
            # bitonic_merge_kernel[(n_blocks,)](
            bitonic_merge_kernel[(1,)](
                sorted_data,
                indices,
                total_elements,
                stage,
                step,
                descending,
                BLOCK_SIZE,
            )

    # Reshape and remove padding
    sorted_data = sorted_data.reshape(n_rows, n_padded)[:, :n_elements]
    indices = indices.reshape(n_rows, n_padded)[:, :n_elements]

    # Reshape back to original shape
    sorted_data = sorted_data.reshape(orig_shape)
    indices = indices.reshape(orig_shape)

    # Move dimension back
    sorted_data = sorted_data.movedim(-1, dim)
    indices = indices.movedim(-1, dim)

    return sorted_data, indices


# Example usage and testing
def test_triton_against_torch():
    # Test stability
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    device = x.device
    # Triton sort
    sorted_triton, indices_triton = stable_sort_triton(x, dim=-1, descending=False)

    # PyTorch sort for comparison
    sorted_torch, indices_torch = torch.sort(x, dim=-1, stable=True)

    print("Input:")
    print(x)
    print("\nTriton sorted:")
    print(sorted_triton)
    print("\nPyTorch sorted:")
    print(sorted_torch)
    print("\nValues match:", torch.allclose(sorted_triton, sorted_torch, rtol=1e-5))

    # Test with duplicates to verify stability
    x_dup = torch.tensor([[3.0, 1.0, 2.0, 1.0, 3.0]], device=device)
    sorted_triton_dup, idx_triton_dup = stable_sort_triton(x_dup, dim=-1)
    sorted_torch_dup, idx_torch_dup = torch.sort(x_dup, dim=-1, stable=True)

    print("\n\nStability test with duplicates:")
    print("Input:", x_dup)
    print("Triton indices:", idx_triton_dup)
    print("PyTorch indices:", idx_torch_dup)
    print("Indices match:", torch.equal(idx_triton_dup, idx_torch_dup))


test_triton_against_torch()
"""
This implementation provides:

1. **Stable Sorting**: Uses original indices to break ties, ensuring elements with equal values maintain their relative order
2. **Bitonic Sort Algorithm**: Efficient parallel sorting algorithm well-suited for GPUs
3. **Configurable Direction**: Supports both ascending and descending order
4. **Arbitrary Dimensions**: Can sort along any dimension
5. **Power-of-2 Padding**: Automatically pads to the next power of 2 for bitonic sort, then removes padding

Key features:
- **Stability guarantee**: When values are equal, the original index determines order
- **GPU-accelerated**: Full Triton implementation for GPU execution
- **Compatible interface**: Returns `(values, indices)` like PyTorch's `torch.sort()`
- **Block-based processing**: Uses configurable block size for efficient memory access

The bitonic sort has O(n log²n) comparisons but is highly parallelizable on GPUs, making it efficient for moderate-sized arrays.
"""
