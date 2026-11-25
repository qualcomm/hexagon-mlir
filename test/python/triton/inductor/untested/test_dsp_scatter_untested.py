# ===- test_dsp_scatter_untested.py ----------------------------------------===
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

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def scatter_kernel(
    input_ptr,  # Pointer to input tensor (source)
    indices_ptr,  # Pointer to indices tensor
    src_ptr,  # Pointer to source values to scatter
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements to scatter
    output_size,  # Size of output tensor (for bounds checking)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Basic scatter kernel - scatters values into output using indices.

    output[indices[i]] = src[i]

    Note: This is a simplified version that may have race conditions
    when multiple threads write to the same location.

    Args:
        input_ptr: Input tensor to copy from (for positions not scattered)
        indices_ptr: Indices specifying where to scatter
        src_ptr: Source values to scatter
        output_ptr: Output tensor
        n_elements: Number of elements to scatter
        output_size: Total size of output
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices and source values
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    src_values = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    # Bounds check
    valid_indices = (indices >= 0) & (indices < output_size)
    scatter_mask = mask & valid_indices

    # Scatter (atomic operation to handle race conditions)
    # Note: Triton's store is not atomic by default, so this may have races
    tl.store(output_ptr + indices, src_values, mask=scatter_mask)


def scatter_triton(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch wrapper for Triton scatter operation (basic version only).

    Args:
        input: Input tensor to scatter into
        dim: Dimension along which to scatter
        index: Index tensor
        src: Source values to scatter

    Returns:
        Output tensor with scattered values
    """
    assert (
        input.device == index.device == src.device
    ), "All tensors must be on same device"
    assert index.dtype in [torch.int32, torch.int64], "Index must be integer type"
    assert index.shape == src.shape, "Index and src must have same shape"

    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim

    # Create output (copy of input)
    output = input.clone()

    # For 1D case only
    if input.dim() == 1:
        n_elements = index.numel()
        output_size = output.size(0)

        BLOCK_SIZE = 1024
        grid = (1,)

        scatter_kernel[grid](
            input,
            index,
            src,
            output,
            n_elements,
            output_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

    # For other dimensions, fall back to PyTorch
    return input.scatter(dim, index, src)


# Testing
def test_scatter():
    """Test scatter implementations."""
    print("Testing scatter Triton kernels...\n")

    # Test 1: Basic 1D scatter
    print("Test 1: Basic 1D scatter")
    input_1d = torch.zeros(5)
    indices_1d = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int64)
    src_1d = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    output_torch = input_1d.scatter(0, indices_1d, src_1d)
    output_triton = scatter_triton(input_1d, 0, indices_1d, src_1d)

    print(f"Input: {input_1d}")
    print(f"Indices: {indices_1d}")
    print(f"Source: {src_1d}")
    print(f"PyTorch: {output_torch}")
    print(f"Triton: {output_triton}")
    assert torch.allclose(output_torch, output_triton), "1D scatter failed!"
    print("✓ 1D scatter passed!\n")


def benchmark_scatter():
    """Benchmark scatter operations."""
    import time

    print("\nBenchmarking scatter operations...\n")

    # Benchmark 1D scatter
    sizes = [1000, 10000, 100000, 1000000]

    print("1D Scatter Benchmark:")
    for size in sizes:
        input = torch.zeros(size)
        indices = torch.randint(0, size, (size,), dtype=torch.int64)
        src = torch.randn(size)

        # Warmup
        for _ in range(10):
            _ = input.scatter(0, indices, src)
            _ = scatter_triton(input, 0, indices, src)

        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = input.scatter(0, indices, src)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = scatter_triton(input, 0, indices, src)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch/time_triton:.2f}x"
        )


if __name__ == "__main__":
    test_scatter()
    benchmark_scatter()
