# ===- test_dsp_gather.py --------------------------------------------------===
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

from .. import dsp, parametrize_dims, parametrize_dtypes

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def gather_1d_kernel(
    input_ptr,  # Pointer to input tensor
    indices_ptr,  # Pointer to indices tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements to gather
    input_size,  # Size of input tensor (for bounds checking)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Basic 1D gather kernel.

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

    # Bounds check (clamp to valid range)
    indices = tl.minimum(tl.maximum(indices, 0), input_size - 1)

    # Gather from input using indices
    gathered_values = tl.load(input_ptr + indices, mask=mask, other=0.0)

    # Store results
    tl.store(output_ptr + offsets, gathered_values, mask=mask)


@triton.jit
def gather_2d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    M,
    N,  # Output dimensions
    input_stride_0,  # Input stride for dim 0
    input_stride_1,  # Input stride for dim 1
    indices_stride_0,  # Indices stride for dim 0
    indices_stride_1,  # Indices stride for dim 1
    output_stride_0,  # Output stride for dim 0
    output_stride_1,  # Output stride for dim 1
    dim,  # Dimension along which to gather (0 or 1)
    input_dim_size,  # Size of input along gather dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D gather kernel with specified dimension.

    For dim=0: output[i, j] = input[indices[i, j], j]
    For dim=1: output[i, j] = input[i, indices[i, j]]

    Args:
        input_ptr: Input tensor
        indices_ptr: Indices tensor (same shape as output)
        output_ptr: Output tensor
        M, N: Dimensions of output/indices
        input_stride_0, input_stride_1: Strides for input
        indices_stride_0, indices_stride_1: Strides for indices
        output_stride_0, output_stride_1: Strides for output
        dim: Dimension to gather along
        input_dim_size: Size of input along gather dimension
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes
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
    indices_offsets = offs_m * indices_stride_0 + offs_n * indices_stride_1
    indices = tl.load(indices_ptr + indices_offsets, mask=mask, other=0)

    # Bounds check and clamp
    indices = tl.minimum(tl.maximum(indices, 0), input_dim_size - 1)

    # Compute input offsets based on gather dimension
    if dim == 0:
        # Gathering along dimension 0: output[i,j] = input[indices[i,j], j]
        input_offsets = indices * input_stride_0 + offs_n * input_stride_1
    else:  # dim == 1
        # Gathering along dimension 1: output[i,j] = input[i, indices[i,j]]
        input_offsets = offs_m * input_stride_0 + indices * input_stride_1

    # Gather values
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Store to output
    output_offsets = offs_m * output_stride_0 + offs_n * output_stride_1
    tl.store(output_ptr + output_offsets, values, mask=mask)


@triton.jit
def gather_3d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    D0,
    D1,
    D2,  # Output dimensions
    input_stride_0,
    input_stride_1,
    input_stride_2,
    indices_stride_0,
    indices_stride_1,
    indices_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    dim,  # Dimension to gather along
    input_dim_size,  # Size of input along gather dimension
    BLOCK_SIZE: tl.constexpr,
):
    """
    3D gather kernel.
    Flattens the computation for simplicity.
    """
    pid = tl.program_id(axis=0)
    n_elements = D0 * D1 * D2

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute 3D indices from flat offset
    d2 = offsets % D2
    d1 = (offsets // D2) % D1
    d0 = offsets // (D1 * D2)

    # Load indices
    indices_flat_offset = (
        d0 * indices_stride_0 + d1 * indices_stride_1 + d2 * indices_stride_2
    )
    indices = tl.load(indices_ptr + indices_flat_offset, mask=mask, other=0)

    # Clamp indices
    indices = tl.minimum(tl.maximum(indices, 0), input_dim_size - 1)

    # Compute input offset based on gather dimension
    if dim == 0:
        input_offset = (
            indices * input_stride_0 + d1 * input_stride_1 + d2 * input_stride_2
        )
    elif dim == 1:
        input_offset = (
            d0 * input_stride_0 + indices * input_stride_1 + d2 * input_stride_2
        )
    else:  # dim == 2
        input_offset = (
            d0 * input_stride_0 + d1 * input_stride_1 + indices * input_stride_2
        )

    # Gather
    values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

    # Store
    output_offset = d0 * output_stride_0 + d1 * output_stride_1 + d2 * output_stride_2
    tl.store(output_ptr + output_offset, values, mask=mask)


@triton.jit
def gather_general_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_elements,  # Total number of elements in output
    dim,  # Dimension to gather along
    input_dim_size,  # Size of input along gather dimension
    ndim,  # Number of dimensions
    input_strides_ptr,  # Pointer to input strides array
    indices_strides_ptr,  # Pointer to indices strides array
    output_strides_ptr,  # Pointer to output strides array
    output_shape_ptr,  # Pointer to output shape array
    BLOCK_SIZE: tl.constexpr,
):
    """
    General N-dimensional gather kernel.
    Works for any number of dimensions.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    flat_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = flat_offsets < n_elements

    # For each flat offset, we need to:
    # 1. Convert to multi-dimensional index
    # 2. Load the gather index
    # 3. Compute input offset
    # 4. Gather and store

    # This is complex in vectorized form, so we process element by element
    # In practice, use specialized kernels for common cases (1D, 2D, 3D)

    # Simplified version - processes one element at a time within the vector
    # For production, you'd want to optimize this further

    for i in range(BLOCK_SIZE):
        if flat_offsets[i] < n_elements:
            flat_idx = flat_offsets[i]

            # Compute indices offset (same as flat_idx for output)
            indices_val = tl.load(indices_ptr + flat_idx)
            indices_val = tl.minimum(tl.maximum(indices_val, 0), input_dim_size - 1)

            # For gather, we need to replace the dim-th coordinate with indices_val
            # This requires decomposing flat_idx, which is expensive
            # Better to use specialized kernels for each ndim

            # Placeholder - load and store
            input_val = tl.load(input_ptr + flat_idx)
            tl.store(output_ptr + flat_idx, input_val)


def gather_triton(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for Triton gather operation.

    Gathers values along an axis specified by dim.

    Args:
        input: Source tensor (N-D)
        dim: Dimension along which to gather
        index: Index tensor (same ndim as input, but can have different shape)

    Returns:
        Gathered tensor with same shape as index

    Examples:
        >>> input = torch.tensor([[1, 2], [3, 4]])
        >>> index = torch.tensor([[0, 0], [1, 0]])
        >>> gather_triton(input, 1, index)
        tensor([[1, 1],
                [4, 3]])
    """
    assert input.device == index.device, "Input and index must be on same device"
    # assert index.dtype in [torch.int32, torch.int64], "Index must be integer type"
    assert index.dtype in [torch.int32], "Index must be integer type"
    assert (
        input.dim() == index.dim()
    ), "Input and index must have same number of dimensions"

    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim

    assert (
        0 <= dim < input.dim()
    ), f"Invalid dimension {dim} for tensor with {input.dim()} dims"

    # Create output tensor with same shape as index
    output = torch.empty_like(index, dtype=input.dtype, device=input.device)

    # Get dimension to gather along
    input_dim_size = input.size(dim)

    # Ensure contiguous for better performance
    input = input.contiguous()
    index = index.contiguous()

    # Handle different dimensionalities
    if input.dim() == 1:
        # 1D case
        n_elements = index.numel()
        input_size = input.size(0)

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        gather_1d_kernel[grid](
            input,
            index,
            output,
            n_elements,
            input_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    elif input.dim() == 2:
        # 2D case
        M, N = index.shape

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )
        # grid = (1,)

        gather_2d_kernel[grid](
            input,
            index,
            output,
            M,
            N,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            output.stride(0),
            output.stride(1),
            dim,
            input_dim_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    elif input.dim() == 3:
        # 3D case
        D0, D1, D2 = index.shape
        n_elements = D0 * D1 * D2

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        gather_3d_kernel[grid](
            input,
            index,
            output,
            D0,
            D1,
            D2,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            index.stride(0),
            index.stride(1),
            index.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            dim,
            input_dim_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    else:
        # For higher dimensions, fall back to PyTorch or use general kernel
        # General kernel is complex, so we fall back for simplicity
        return torch.gather(input, dim, index)

    return output


@triton.jit
def take_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    input_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Take elements from flattened input using flat indices.
    Equivalent to input.flatten()[indices].
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Clamp to valid range
    indices = tl.minimum(tl.maximum(indices, 0), input_size - 1)

    # Gather
    values = tl.load(input_ptr + indices, mask=mask, other=0.0)

    # Store
    tl.store(output_ptr + offsets, values, mask=mask)


def take_triton(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Take elements from input using flat indices.

    Args:
        input: Input tensor (any shape)
        index: Indices into flattened input

    Returns:
        Gathered values
    """
    input_flat = input.flatten()
    index_flat = index.flatten()

    n_elements = index_flat.numel()
    input_size = input_flat.numel()

    output = torch.empty_like(index_flat, dtype=input.dtype, device=input.device)

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    take_kernel[grid](
        input_flat,
        index_flat,
        output,
        n_elements,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(index.shape)


@triton.jit
def index_select_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_output_elements,
    n_indices,
    stride_before_dim,  # Product of dimensions before selected dim
    stride_after_dim,  # Product of dimensions after selected dim
    input_dim_size,  # Size of input along selected dimension
    BLOCK_SIZE: tl.constexpr,
):
    """
    Index select - select specific slices along a dimension.

    Similar to gather but selects entire slices rather than individual elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_output_elements

    # Decompose output offset into (before, index, after)
    after_idx = offsets % stride_after_dim
    index_idx = (offsets // stride_after_dim) % n_indices
    before_idx = offsets // (n_indices * stride_after_dim)

    # Load the actual index value
    selected_idx = tl.load(indices_ptr + index_idx, mask=mask, other=0)
    selected_idx = tl.minimum(tl.maximum(selected_idx, 0), input_dim_size - 1)

    # Compute input offset
    input_offset = (
        before_idx * input_dim_size * stride_after_dim
        + selected_idx * stride_after_dim
        + after_idx
    )

    # Gather and store
    value = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def index_select_triton(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """
    Select specific indices along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to select along
        index: 1D tensor of indices to select

    Returns:
        Tensor with selected slices
    """
    assert index.dim() == 1, "Index must be 1D for index_select"

    if dim < 0:
        dim = input.dim() + dim

    # Compute output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.numel()

    # Compute strides
    stride_before_dim = 1
    for i in range(dim + 1, input.dim()):
        stride_before_dim *= input.size(i)

    stride_after_dim = stride_before_dim
    if dim < input.dim() - 1:
        stride_after_dim = 1
        for i in range(dim + 1, input.dim()):
            stride_after_dim *= input.size(i)

    n_output_elements = torch.prod(torch.tensor(output_shape)).item()
    n_indices = index.numel()
    input_dim_size = input.size(dim)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    input = input.contiguous()
    index = index.contiguous()

    BLOCK_SIZE = 16
    grid = (triton.cdiv(n_output_elements, BLOCK_SIZE),)
    # grid = (1,)

    index_select_kernel[grid](
        input,
        index,
        output,
        n_output_elements,
        n_indices,
        stride_before_dim,
        stride_after_dim,
        input_dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_gather_1d():
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int32)

    output = gather_triton(x, 0, indices)
    reference = torch.gather(x, 0, indices.to(torch.int64))

    assert torch.allclose(output, reference)


@dsp
@parametrize_dims(0, 1)
def test_gather_2d(dim):
    x = torch.randn(5, 5)
    indices = torch.randint(0, 5, (3, 4), dtype=torch.int32)

    output = gather_triton(x, dim, indices)
    reference = torch.gather(x, dim, indices.to(torch.int64))

    assert torch.allclose(output, reference)


@dsp
@parametrize_dims(0, 1, 2)
def test_gather_3d(dim):
    x = torch.randn(5, 5, 5)
    indices = torch.randint(0, 5, (4, 3, 5), dtype=torch.int32)

    output = gather_triton(x, dim, indices)
    reference = torch.gather(x, dim, indices.to(torch.int64))

    assert torch.allclose(output, reference)


@dsp
def test_take():
    x = torch.randn(20)
    indices = torch.tensor([5, 10, 15, 2, 18], dtype=torch.int32)

    output = take_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


@dsp
def test_gather_index_select():
    x = torch.randn(5, 10)
    indices = torch.tensor([0, 2, 4], dtype=torch.int32)

    output = index_select_triton(x, 0, indices)
    reference = torch.index_select(x, 0, indices.to(torch.int64))

    assert torch.allclose(output, reference)


# TODO: This test is a no-op, fix it.

# @dsp
# def test_out_of_bounds():
#     x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
#     # indices = torch.tensor([0, 10, -1, 2, 100], dtype=torch.int64)
#     indices = torch.tensor([0, 10, -1, 2, 100], dtype=torch.int32)
#
#     # PyTorch gather doesn't clamp - it errors. Our kernel clamps.
#     # So we test that our kernel doesn't crash and produces reasonable output
#     output = gather_triton(x, 0, indices)


@dsp
@parametrize_dtypes(torch.float16, torch.float32)
def test_different_dtypes(dtype):
    x = torch.randn(100, dtype=dtype)
    indices = torch.randint(0, 100, (50,), dtype=torch.int32)

    output = gather_triton(x, 0, indices)
    reference = torch.gather(x, 0, indices.to(torch.int64))

    assert torch.allclose(output, reference, rtol=1e-3)


@dsp
def test_large_tensor():
    x = torch.randn(1000, 500)
    indices = torch.randint(0, 500, (1000, 300), dtype=torch.int32)

    output = gather_triton(x, 1, indices)
    reference = torch.gather(x, 1, indices.to(torch.int64))

    assert torch.allclose(output, reference)
