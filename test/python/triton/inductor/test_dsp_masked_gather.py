# ===- test_dsp_masked_gather.py --------------------------------------------===
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

from .. import dsp, parametrize_dims

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def gather_kernel(
    input_ptr,  # Pointer to input tensor
    indices_ptr,  # Pointer to indices tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements to gather
    input_size,  # Size of input tensor (for bounds checking)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Basic gather kernel - gathers elements from input using indices.

    output[i] = input[indices[i]]

    Args:
        input_ptr: Source tensor to gather from
        indices_ptr: Indices specifying which elements to gather
        output_ptr: Output tensor to store gathered values
        n_elements: Number of elements to gather
        input_size: Total size of input (for bounds checking)
        BLOCK_SIZE: Number of elements per block
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Bounds check (optional but safer)
    valid_indices = (indices >= 0) & (indices < input_size)
    gather_mask = mask & valid_indices

    # Gather from input using indices
    gathered_values = tl.load(input_ptr + indices, mask=gather_mask, other=0.0)

    # Store results
    tl.store(output_ptr + offsets, gathered_values, mask=mask)


@triton.jit
def masked_gather_kernel(
    input_ptr,  # Pointer to input tensor
    mask_ptr,  # Pointer to boolean mask
    indices_ptr,  # Pointer to gather indices
    output_ptr,  # Pointer to output tensor
    default_value,  # Value to use where mask is False
    n_elements,  # Total number of elements
    input_size,  # Size of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    """
    Masked gather kernel - gathers elements only where mask is True.

    output[i] = input[indices[i]] if mask[i] else default_value

    Args:
        input_ptr: Source tensor
        mask_ptr: Boolean mask (True = gather, False = use default)
        indices_ptr: Gather indices
        output_ptr: Output tensor
        default_value: Value used where mask is False
        n_elements: Number of output elements
        input_size: Size of input tensor
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load mask
    mask_gather = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)

    # Bounds check
    valid_indices = (indices >= 0) & (indices < input_size)
    gather_mask = mask_valid & mask_gather & valid_indices

    # Gather values
    gathered = tl.load(input_ptr + indices, mask=gather_mask, other=default_value)

    # Use default value where mask is False
    output = tl.where(mask_gather, gathered, default_value)

    # Store
    tl.store(output_ptr + offsets, output, mask=mask_valid)


@triton.jit
def gather_2d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    M,
    N,  # Output dimensions
    input_stride_0,  # Input stride for dim 0
    indices_stride_0,  # Indices stride for dim 0
    output_stride_0,  # Output stride for dim 0
    dim,  # Dimension along which to gather (0 or 1)
    input_dim_size,  # Size of input along gather dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D gather kernel with specified dimension.

    For dim=0: output[i, j] = input[indices[i, j], j]
    For dim=1: output[i, j] = input[i, indices[i, j]]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m & mask_n

    # Load indices
    indices_offsets = offs_m * indices_stride_0 + offs_n
    indices = tl.load(indices_ptr + indices_offsets, mask=mask, other=0)

    # Bounds check
    valid_indices = (indices >= 0) & (indices < input_dim_size)

    # Compute input offsets based on gather dimension
    if dim == 0:
        # Gathering along dimension 0
        input_offsets = indices * input_stride_0 + offs_n
    else:  # dim == 1
        # Gathering along dimension 1
        input_offsets = offs_m * input_stride_0 + indices

    # Gather
    gather_mask = mask & valid_indices
    values = tl.load(input_ptr + input_offsets, mask=gather_mask, other=0.0)

    # Store
    output_offsets = offs_m * output_stride_0 + offs_n
    tl.store(output_ptr + output_offsets, values, mask=mask)


def gather_triton(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for Triton gather operation.

    Gathers values along an axis specified by dim.

    Args:
        input: Source tensor
        dim: Dimension along which to gather
        index: Index tensor (same shape as output, contains indices into input's dim)

    Returns:
        Gathered tensor with same shape as index
    """
    assert input.device == index.device, "Input and index must be on same device"
    assert index.dtype in [torch.int32, torch.int64], "Index must be integer type"

    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim

    assert (
        0 <= dim < input.dim()
    ), f"Invalid dimension {dim} for tensor with {input.dim()} dims"

    # Create output tensor with same shape as index
    output = torch.empty_like(index, dtype=input.dtype)

    # For 1D case
    if input.dim() == 1:
        n_elements = index.numel()
        input_size = input.size(0)

        BLOCK_SIZE = 1024
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        gather_kernel[grid](
            input,
            index,
            output,
            n_elements,
            input_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

    # For 2D case
    if input.dim() == 2 and index.dim() == 2:
        M, N = index.shape
        input_dim_size = input.size(dim)

        # Ensure contiguous
        input = input.contiguous()
        index = index.contiguous()

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16

        # grid = (
        #    triton.cdiv(M, BLOCK_SIZE_M),
        #    triton.cdiv(N, BLOCK_SIZE_N),
        # )
        grid = (1,)

        gather_2d_kernel[grid](
            input,
            index,
            output,
            M,
            N,
            input.stride(0),
            index.stride(0),
            output.stride(0),
            dim,
            input_dim_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return output

    # For general ND case, flatten and reshape
    # This is a simplified implementation
    input_flat = input.flatten()
    index_flat = index.flatten()

    # Compute flat indices
    # This is complex for general ND - using PyTorch for now
    output_flat = torch.gather(input, dim, index)
    return output_flat


def masked_gather_triton(
    input: torch.Tensor,
    mask: torch.Tensor,
    index: torch.Tensor,
    default_value: float = 0.0,
) -> torch.Tensor:
    """
    Masked gather operation.

    Args:
        input: Source tensor (1D)
        mask: Boolean mask (same shape as index)
        index: Indices to gather
        default_value: Value to use where mask is False

    Returns:
        Gathered values (same shape as index)
    """
    assert input.dim() == 1, "This implementation supports 1D input"
    assert mask.shape == index.shape, "Mask and index must have same shape"
    assert mask.dtype == torch.bool, "Mask must be boolean"

    input_flat = input.flatten()
    mask_flat = mask.flatten()
    index_flat = index.flatten()

    n_elements = index_flat.numel()
    input_size = input_flat.numel()

    output = torch.empty_like(index_flat, dtype=input.dtype)

    BLOCK_SIZE = 1024
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    masked_gather_kernel[grid](
        input_flat,
        mask_flat,
        index_flat,
        output,
        default_value,
        n_elements,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(index.shape)


@triton.jit
def gather_nd_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    n_dims,  # Number of dimensions
    strides_ptr,  # Input strides
    dims_ptr,  # Input dimensions (for bounds checking)
    BLOCK_SIZE: tl.constexpr,
):
    """
    N-dimensional gather kernel.
    Indices shape: [n_elements, n_dims]
    Each row of indices specifies one element to gather.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # For each element, compute flat input offset from multi-dim indices
    # This requires loading multiple index values per element
    # Simplified for demonstration

    # Load first dimension index as example
    idx_offset = offsets * n_dims
    idx_0 = tl.load(indices_ptr + idx_offset, mask=mask, other=0)

    # For full implementation, need to:
    # 1. Load all dimension indices
    # 2. Compute flat offset using strides
    # 3. Gather from input

    # Placeholder - full implementation would iterate over dimensions
    flat_offset = idx_0
    values = tl.load(input_ptr + flat_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def take_along_dim_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    M,
    N,
    input_stride_m,
    input_stride_n,
    indices_stride_m,
    indices_stride_n,
    output_stride_m,
    output_stride_n,
    dim,  # 0 for rows, 1 for columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Take along dimension (similar to np.take_along_axis).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m & mask_n

    # Load indices
    indices_offsets = offs_m * indices_stride_m + offs_n * indices_stride_n
    indices = tl.load(indices_ptr + indices_offsets, mask=mask, other=0)

    # Compute input offsets based on dimension
    if dim == 0:
        # Take along rows (dim 0)
        input_offsets = indices * input_stride_m + offs_n * input_stride_n
    else:
        # Take along columns (dim 1)
        input_offsets = offs_m * input_stride_m + indices * input_stride_n

    # Gather
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Store
    output_offsets = offs_m * output_stride_m + offs_n * output_stride_n
    tl.store(output_ptr + output_offsets, values, mask=mask)


def take_along_dim_triton(
    input: torch.Tensor, dim: int, indices: torch.Tensor
) -> torch.Tensor:
    """
    Take values from input along dimension dim using indices.
    Similar to numpy's take_along_axis.

    Args:
        input: Input tensor (2D)
        dim: Dimension to take along (0 or 1)
        indices: Indices tensor (same shape as output)

    Returns:
        Gathered tensor
    """
    assert input.dim() == 2, "This implementation supports 2D tensors"
    assert indices.dim() == 2, "Indices must be 2D"

    input = input.contiguous()
    indices = indices.contiguous()

    M, N = indices.shape
    output = torch.empty_like(indices, dtype=input.dtype)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16

    # grid = (
    #    triton.cdiv(M, BLOCK_SIZE_M),
    #    triton.cdiv(N, BLOCK_SIZE_N),
    # )
    grid = (1,)

    take_along_dim_kernel[grid](
        input,
        indices,
        output,
        M,
        N,
        input.stride(0),
        input.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_masked_gather_1d():
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int32)

    reference = torch.gather(x, 0, indices.long())
    output = gather_triton(x, 0, indices)

    assert torch.allclose(output, reference)


@dsp
@parametrize_dims(0, 1)
def test_masked_gather_2d(dim):
    x = torch.randn(5, 5)
    indices = torch.randint(0, 5, (3, 4), dtype=torch.int32)

    reference = torch.gather(x, dim, indices.to(torch.int64))
    output = gather_triton(x, dim, indices)

    assert torch.allclose(reference, output)


@dsp
def test_masked_gather():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)
    indices = torch.tensor([4, 2, 0, 3, 1], dtype=torch.int32)
    default = -99.0

    output = masked_gather_triton(x, mask, indices, default)
    expected = torch.tensor([5.0, -99.0, 1.0, -99.0, 2.0])

    assert torch.allclose(output, expected)


@dsp
def test_take_along_dim():
    x = torch.randn(4, 5)
    indices = torch.randint(0, 4, (4, 5), dtype=torch.int32)

    output = take_along_dim_triton(x, 0, indices)
    reference = torch.gather(x, 0, indices.to(torch.int64))

    assert torch.allclose(output, reference)


def benchmark_gather():
    """Benchmark gather operations."""
    import time

    print("\nBenchmarking gather operations...\n")

    # Benchmark 1D gather
    sizes = [1000, 10000, 100000, 1000000]

    print("1D Gather Benchmark:")
    for size in sizes:
        input = torch.randn(size)
        indices = torch.randint(0, size, (size,), dtype=torch.int64)

        # Warmup
        for _ in range(10):
            _ = torch.gather(input, 0, indices)
            _ = gather_triton(input, 0, indices)


        # Benchmark
        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = torch.gather(input, 0, indices)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = gather_triton(input, 0, indices)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark 2D gather
    print("\n2D Gather Benchmark:")
    shapes = [(100, 100), (512, 512), (1024, 1024), (2048, 2048)]

    for M, N in shapes:
        input = torch.randn(M, N)
        indices = torch.randint(0, M, (M, N), dtype=torch.int64)

        # Warmup
        for _ in range(10):
            _ = torch.gather(input, 0, indices)
            _ = gather_triton(input, 0, indices)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = torch.gather(input, 0, indices)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = gather_triton(input, 0, indices)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Shape {M}x{N}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark_gather()
