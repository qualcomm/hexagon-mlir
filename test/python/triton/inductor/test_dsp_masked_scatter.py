# ===- test_dsp_masked_scatter.py -------------------------------------------===
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

from .. import dsp, parametrize_dtypes

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def masked_scatter_kernel(
    input_ptr,  # Pointer to input tensor
    mask_ptr,  # Pointer to mask tensor (boolean)
    source_ptr,  # Pointer to source tensor (values to scatter)
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/mask
    source_size,  # Total number of elements in source
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for masked_scatter operation.

    Copies values from source into self (output) at positions where mask is True.
    Uses values from source sequentially, regardless of the shape of mask.

    Args:
        input_ptr: Input tensor to copy from where mask is False
        mask_ptr: Boolean mask tensor
        source_ptr: Source tensor with values to scatter
        output_ptr: Output tensor
        n_elements: Number of elements in input/mask/output
        source_size: Number of elements in source
        BLOCK_SIZE: Elements processed per block
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load input and mask
    input_val = tl.load(input_ptr + offsets, mask=mask_valid, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

    # For each element, determine if we should use source or input
    # We need to know the cumulative count of True values before this position
    # This requires a different approach - see the two-pass implementation below

    # Note: This is a simplified version. Full implementation requires
    # computing prefix sum of mask to know source indices.
    # See the two-pass version below for correct implementation.

    output_val = tl.where(mask_val, input_val, input_val)  # Placeholder
    tl.store(output_ptr + offsets, output_val, mask=mask_valid)


@triton.jit
def count_true_kernel(
    mask_ptr,
    count_ptr,  # Output: cumulative count at each position
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute cumulative count of True values in mask (prefix sum).
    This tells us which source element to use at each position.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load mask values
    mask_val = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

    # Convert bool to int for counting
    mask_int = mask_val.to(tl.int32)

    # This is a simplified version - proper prefix sum requires
    # multiple passes or shuffle operations
    # For now, we'll use the CPU-based approach
    tl.store(count_ptr + offsets, mask_int, mask=mask_valid)


@triton.jit
def masked_scatter_with_indices_kernel(
    input_ptr,
    mask_ptr,
    source_ptr,
    indices_ptr,  # Prefix sum of mask (which source element to use)
    output_ptr,
    n_elements,
    source_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Masked scatter kernel using precomputed indices.

    The indices array contains the cumulative count of True values
    up to (but not including) each position.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load values
    input_val = tl.load(input_ptr + offsets, mask=mask_valid, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask_valid, other=0) != 0
    source_idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)

    # Load from source using computed indices
    # Only load if mask is True and index is valid
    source_mask = mask_val & (source_idx < source_size)
    source_val = tl.load(source_ptr + source_idx, mask=source_mask, other=0.0)

    # Select between input and source based on mask
    output_val = tl.where(mask_val, source_val, input_val)

    tl.store(output_ptr + offsets, output_val, mask=mask_valid)


def masked_scatter_triton(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch wrapper for Triton masked_scatter.

    Copies elements from source into input at positions where mask is True.
    Source values are used sequentially.

    Args:
        input: Input tensor (any shape)
        mask: Boolean mask tensor (same shape as input)
        source: Source tensor (1D or any shape, values used sequentially)

    Returns:
        Output tensor with scattered values
    """
    assert input.shape == mask.shape, "Input and mask must have same shape"
    assert (
        input.device == mask.device == source.device
    ), "All tensors must be on same device"
    assert mask.dtype == torch.bool, "Mask must be boolean"

    # Flatten tensors for easier processing
    input_flat = input.flatten()
    mask_flat = mask.flatten().to(torch.int8)
    source_flat = source.flatten()

    n_elements = input_flat.numel()
    source_size = source_flat.numel()

    # Check that source has enough elements
    n_true = mask_flat.sum().item()
    if n_true > source_size:
        raise RuntimeError(
            f"Number of True values in mask ({n_true}) exceeds source size ({source_size})"
        )

    # Create output tensor
    output = input_flat.clone()

    # Compute prefix sum (cumulative count) of mask on CPU/GPU
    # This tells us which source index to use at each position
    mask_int = mask_flat.to(torch.int32)
    indices = (torch.cumsum(mask_int, dim=0) - mask_int).to(torch.int32)

    # Launch kernel
    BLOCK_SIZE = 1024
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    masked_scatter_with_indices_kernel[grid](
        input_flat,
        mask_flat,
        source_flat,
        indices,
        output,
        n_elements,
        source_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(input.shape)


# Optimized version with 2D blocking
@triton.jit
def masked_scatter_2d_kernel(
    input_ptr,
    mask_ptr,
    source_ptr,
    indices_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
    source_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D version with better memory access patterns.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]

    offsets = offs_m * stride_m + offs_n * stride_n

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_valid = mask_m & mask_n

    # Load input, mask, and indices
    input_val = tl.load(input_ptr + offsets, mask=mask_valid, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)
    source_idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)

    # Load from source
    source_mask = mask_valid & mask_val & (source_idx < source_size)
    source_val = tl.load(source_ptr + source_idx, mask=source_mask, other=0.0)

    # Scatter
    output_val = tl.where(mask_val, source_val, input_val)
    tl.store(output_ptr + offsets, output_val, mask=mask_valid)


def masked_scatter_triton_2d(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    2D optimized version for better memory coalescing.
    """
    assert input.shape == mask.shape
    assert input.is_contiguous() and mask.is_contiguous()

    # Reshape to 2D
    if input.dim() == 1:
        M, N = 1, input.shape[0]
        input_2d = input.reshape(M, N)
        mask_2d = mask.reshape(M, N)
    elif input.dim() == 2:
        M, N = input.shape
        input_2d = input
        mask_2d = mask
    else:
        M = input.shape[0]
        N = input.numel() // M
        input_2d = input.reshape(M, N)
        mask_2d = mask.reshape(M, N)

    source_flat = source.flatten()
    source_size = source_flat.numel()

    # Compute indices (prefix sum)
    mask_flat = mask_2d.flatten()
    mask_int = mask_flat.to(torch.int32)
    indices = torch.cumsum(mask_int, dim=0) - mask_int
    indices_2d = indices.reshape(M, N)

    output = input_2d.clone()

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    # grid = (
    #    triton.cdiv(M, BLOCK_SIZE_M),
    #    triton.cdiv(N, BLOCK_SIZE_N),
    # )
    grid = (1,)

    masked_scatter_2d_kernel[grid](
        input_2d,
        mask_2d,
        source_flat,
        indices_2d,
        output,
        M,
        N,
        input_2d.stride(0),
        input_2d.stride(1),
        source_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output.reshape(input.shape)


# In-place version
def masked_scatter_triton_(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    In-place version of masked_scatter.
    Modifies input tensor directly.
    """
    assert input.shape == mask.shape
    assert input.device == mask.device == source.device

    input_flat = input.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    n_elements = input_flat.numel()
    source_size = source_flat.numel()

    n_true = mask_flat.sum().item()
    if n_true > source_size:
        raise RuntimeError(
            f"Not enough source elements: need {n_true}, have {source_size}"
        )

    # Compute indices
    mask_int = mask_flat.to(torch.int32)
    indices = (torch.cumsum(mask_int, dim=0) - mask_int).to(torch.int32)

    BLOCK_SIZE = 1024
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    masked_scatter_with_indices_kernel[grid](
        input_flat,
        mask_flat,
        source_flat,
        indices,
        input_flat,  # Write back to input (in-place)
        n_elements,
        source_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return input


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_masked_scatter_1d():
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([False, True, False, True, False], dtype=torch.bool)
    source = torch.tensor([10.0, 20.0, 30.0])

    output = masked_scatter_triton(input, mask, source)
    reference = input.masked_scatter(mask, source)

    assert torch.allclose(reference, output)


@dsp
def test_masked_scatter_2d():
    input = torch.randn(4, 6)
    mask = torch.randint(0, 2, (4, 6), dtype=torch.bool)
    n_true = mask.sum().item()
    source = torch.randn(n_true)

    output = masked_scatter_triton(input, mask, source)
    reference = input.masked_scatter(mask, source)

    assert torch.allclose(output, reference)


@dsp
def test_masked_scatter_large():
    input = torch.randn(4, 256)
    mask = torch.randint(0, 2, (4, 256), dtype=torch.bool)
    n_true = mask.sum().item()
    source = torch.randn(n_true)

    output = masked_scatter_triton(input, mask, source)
    reference = input.masked_scatter(mask, source)

    assert torch.allclose(output, reference, rtol=1e-5)


@dsp
def test_masked_scatter_inplace():
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([False, True, False, True, False], dtype=torch.bool)
    source = torch.tensor([10.0, 20.0, 30.0])

    reference = input.clone()
    output = input.clone()

    reference.masked_scatter_(mask, source)
    masked_scatter_triton_(output, mask, source)

    assert torch.allclose(output, reference)


@dsp
@parametrize_dtypes(torch.float16, torch.float32)
def test_masked_scatter_dtypes(dtype):
    input = torch.randn(100, dtype=dtype)
    mask = torch.randint(0, 2, (100,), dtype=torch.bool)
    n_true = mask.sum().item()
    source = torch.randn(n_true, dtype=dtype)

    output = masked_scatter_triton(input, mask, source)
    reference = input.masked_scatter(mask, source)

    assert torch.allclose(output, reference, rtol=1e-3)


def benchmark_masked_scatter():
    """Benchmark Triton vs PyTorch implementation."""
    import time

    print("\nBenchmarking masked_scatter...")

    sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    sparsities = [0.1, 0.5, 0.9]  # Fraction of True values in mask

    for M, N in sizes:
        for sparsity in sparsities:
            input = torch.randn(M, N)
            mask = torch.rand(M, N) < sparsity
            n_true = mask.sum().item()
            source = torch.randn(n_true)

            # Warmup
            for _ in range(10):
                _ = input.masked_scatter(mask, source)
                _ = masked_scatter_triton(input, mask, source)


            # Benchmark PyTorch
            n_iters = 100
            start = time.time()
            for _ in range(n_iters):
                result_torch = input.masked_scatter(mask, source)
            time_torch = (time.time() - start) / n_iters * 1000

            # Benchmark Triton
            start = time.time()
            for _ in range(n_iters):
                result_triton = masked_scatter_triton(input, mask, source)
            time_triton = (time.time() - start) / n_iters * 1000

            print(f"\nSize: {M}x{N}, Sparsity: {sparsity:.1f} ({n_true} elements)")
            print(f" PyTorch: {time_torch:.4f} ms")
            print(f" Triton: {time_triton:.4f} ms")
            print(f" Speedup: {time_torch / time_triton:.2f}x")


if __name__ == "__main__":
    benchmark_masked_scatter()
