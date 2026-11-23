# ===- test_dsp_index.py ----------------------------------------------------===
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
def index_1d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    input_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D indexing: output = input[indices]

    Args:
        input_ptr: Input tensor (1D)
        indices_ptr: Index tensor (1D)
        output_ptr: Output tensor
        n_elements: Number of elements to index
        input_size: Size of input (for bounds checking)
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Handle negative indices: if idx < 0, convert to idx + input_size
    indices = tl.where(indices < 0, indices + input_size, indices)

    # Clamp to valid range [0, input_size-1]
    indices = tl.minimum(tl.maximum(indices, 0), input_size - 1)

    # Gather from input
    values = tl.load(input_ptr + indices, mask=mask, other=0.0)

    # Store to output
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def index_2d_single_dim_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_indices,
    dim_size,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    dim,  # Which dimension to index (0 or 1)
    input_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D indexing along one dimension.

    For dim=0: output = input[indices, :]
    For dim=1: output = input[:, indices]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Total elements = n_indices * dim_size
    n_total = n_indices * dim_size
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_total

    # Decompose flat offset
    if dim == 0:
        # Indexing rows: output[i, j] = input[indices[i], j]
        col = offsets % dim_size
        idx_num = offsets // dim_size

        # Load index
        row_idx = tl.load(indices_ptr + idx_num, mask=mask, other=0)

        # Handle negative indices
        row_idx = tl.where(row_idx < 0, row_idx + input_dim_size, row_idx)
        row_idx = tl.minimum(tl.maximum(row_idx, 0), input_dim_size - 1)

        # Compute input and output offsets
        input_offset = row_idx * input_stride_0 + col * input_stride_1
        output_offset = idx_num * output_stride_0 + col * output_stride_1
    else:
        # Indexing columns: output[i, j] = input[i, indices[j]]
        row = offsets % dim_size
        idx_num = offsets // dim_size

        # Load index
        col_idx = tl.load(indices_ptr + idx_num, mask=mask, other=0)

        # Handle negative indices
        col_idx = tl.where(col_idx < 0, col_idx + input_dim_size, col_idx)
        col_idx = tl.minimum(tl.maximum(col_idx, 0), input_dim_size - 1)

        # Compute offsets
        input_offset = row * input_stride_0 + col_idx * input_stride_1
        output_offset = row * output_stride_0 + idx_num * output_stride_1

    # Load and store
    value = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, value, mask=mask)


@triton.jit
def index_2d_dual_dim_kernel(
    input_ptr,
    indices_0_ptr,
    indices_1_ptr,
    output_ptr,
    n_elements,
    input_stride_0,
    input_stride_1,
    input_dim_0_size,
    input_dim_1_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D indexing with indices for both dimensions.

    output[i] = input[indices_0[i], indices_1[i]]

    Both index arrays must be broadcastable to same shape.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices for both dimensions
    idx_0 = tl.load(indices_0_ptr + offsets, mask=mask, other=0)
    idx_1 = tl.load(indices_1_ptr + offsets, mask=mask, other=0)

    # Handle negative indices
    idx_0 = tl.where(idx_0 < 0, idx_0 + input_dim_0_size, idx_0)
    idx_1 = tl.where(idx_1 < 0, idx_1 + input_dim_1_size, idx_1)

    # Clamp to valid range
    idx_0 = tl.minimum(tl.maximum(idx_0, 0), input_dim_0_size - 1)
    idx_1 = tl.minimum(tl.maximum(idx_1, 0), input_dim_1_size - 1)

    # Compute input offset
    input_offset = idx_0 * input_stride_0 + idx_1 * input_stride_1

    # Load and store
    value = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def index_boolean_mask_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    write_indices_ptr,  # Prefix sum of mask
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Boolean mask indexing: output = input[mask]

    Requires precomputed prefix sum for write positions.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load mask
    mask_bool = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

    # Load input values
    input_val = tl.load(input_ptr + offsets, mask=mask_valid, other=0.0)

    # Load write position
    write_idx = tl.load(write_indices_ptr + offsets, mask=mask_valid, other=0)

    # Store to output only where mask is True
    tl.store(output_ptr + write_idx, input_val, mask=mask_valid & mask_bool)


@triton.jit
def index_slice_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    start,
    step,
    input_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Slice indexing: output = input[start:stop:step]

    Args:
        input_ptr: Input tensor
        output_ptr: Output tensor
        n_elements: Number of elements in output
        start: Starting index
        step: Step size
        input_stride: Stride of input tensor
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute input indices: start + i * step
    input_indices = start + offsets * step
    input_offset = input_indices * input_stride

    # Load and store
    values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


def index_triton(input: torch.Tensor, indices) -> torch.Tensor:
    """
    PyTorch wrapper for Triton advanced indexing.

    Supports:
    - Integer indexing
    - Slice indexing
    - Tensor indexing
    - Boolean mask indexing
    - Mixed indexing

    Args:
        input: Input tensor
        indices: Index specification (int, slice, tensor, tuple of indices)

    Returns:
        Indexed tensor

    Examples:
        >>> x = torch.randn(10, 5)
        >>> index_triton(x, 3) # Single integer
        >>> index_triton(x, torch.tensor([0, 2, 5])) # Integer tensor
        >>> index_triton(x, slice(0, 5, 2)) # Slice
        >>> index_triton(x, (torch.tensor([0, 2]), torch.tensor([1, 3]))) # Multi-dim
    """

    # Normalize indices to tuple
    if not isinstance(indices, tuple):
        indices = (indices,)

    # Handle different indexing cases

    # Case 1: Single integer index
    if len(indices) == 1 and isinstance(indices[0], int):
        idx = indices[0]
        if idx < 0:
            idx += input.size(0)
        return input[idx].clone()

    # Case 2: Single tensor index (1D)
    if len(indices) == 1 and isinstance(indices[0], torch.Tensor):
        index_tensor = indices[0].contiguous()

        if index_tensor.dtype == torch.bool:
            # Boolean mask indexing
            return boolean_index_triton(input, index_tensor)

        if input.dim() == 1:
            # 1D input, 1D index
            n_elements = index_tensor.numel()
            input_size = input.size(0)

            output = torch.empty(
                index_tensor.shape, dtype=input.dtype, device=input.device
            )

            index_flat = index_tensor.flatten()
            output_flat = output.flatten()

            BLOCK_SIZE = 1024
            # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            grid = (1,)

            index_1d_kernel[grid](
                input,
                index_flat,
                output_flat,
                n_elements,
                input_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return output

        elif input.dim() == 2:
            # 2D input, indexing first dimension
            n_indices = index_tensor.numel()
            dim_size = input.size(1)

            output_shape = list(index_tensor.shape) + [input.size(1)]
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

            index_flat = index_tensor.flatten()
            output_2d = output.view(-1, dim_size)

            BLOCK_SIZE = 256
            # grid = lambda meta: (triton.cdiv(n_indices * dim_size, meta['BLOCK_SIZE']),)
            grid = (1,)

            index_2d_single_dim_kernel[grid](
                input,
                index_flat,
                output_2d,
                n_indices,
                dim_size,
                input.stride(0),
                input.stride(1),
                output_2d.stride(0),
                output_2d.stride(1),
                0,  # dim=0
                input.size(0),
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return output

    # Case 3: Single slice
    if len(indices) == 1 and isinstance(indices[0], slice):
        s = indices[0]
        start, stop, step = s.start, s.stop, s.step

        # Normalize slice parameters
        if start is None:
            start = 0
        if start < 0:
            start += input.size(0)

        if stop is None:
            stop = input.size(0)
        if stop < 0:
            stop += input.size(0)

        if step is None:
            step = 1

        # Compute output size
        n_elements = max(0, (stop - start + step - 1) // step)

        if input.dim() == 1:
            output = torch.empty(n_elements, dtype=input.dtype, device=input.device)

            BLOCK_SIZE = 1024
            # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            grid = (1,)

            index_slice_kernel[grid](
                input,
                output,
                n_elements,
                start,
                step,
                1,  # stride for 1D
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return output
        else:
            # For multi-dim, use PyTorch
            return input[indices[0]].clone()

    # Case 4: Multiple tensor indices
    if len(indices) == 2 and all(isinstance(idx, torch.Tensor) for idx in indices):
        idx_0, idx_1 = indices

        # Check for boolean masks
        if idx_0.dtype == torch.bool or idx_1.dtype == torch.bool:
            return input[indices].clone()

        if input.dim() == 2:
            # Broadcast indices to same shape
            idx_0_bc, idx_1_bc = torch.broadcast_tensors(idx_0, idx_1)
            n_elements = idx_0_bc.numel()

            output = torch.empty(idx_0_bc.shape, dtype=input.dtype, device=input.device)

            idx_0_flat = idx_0_bc.flatten().contiguous()
            idx_1_flat = idx_1_bc.flatten().contiguous()
            output_flat = output.flatten()

            BLOCK_SIZE = 256
            # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            grid = (1,)

            index_2d_dual_dim_kernel[grid](
                input,
                idx_0_flat,
                idx_1_flat,
                output_flat,
                n_elements,
                input.stride(0),
                input.stride(1),
                input.size(0),
                input.size(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return output

    # Case 5: Mixed indexing (integers, slices, tensors)
    # This is complex - fall back to PyTorch
    return input[indices].clone()


def boolean_index_triton(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Boolean mask indexing: input[mask]

    Args:
        input: Input tensor
        mask: Boolean mask (same shape as input)

    Returns:
        1D tensor with selected elements
    """
    assert input.shape == mask.shape, "Input and mask must have same shape"
    assert mask.dtype == torch.bool, "Mask must be boolean"

    # Flatten
    input_flat = input.flatten()
    mask_flat = mask.flatten()

    # Compute prefix sum for write positions
    mask_int = mask_flat.to(torch.int32)
    write_indices = (torch.cumsum(mask_int, dim=0) - mask_int).to(torch.int32)

    # Count True values
    n_true = mask_flat.sum().item()

    if n_true == 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    # Allocate output
    output = torch.empty(n_true, dtype=input.dtype, device=input.device)

    n_elements = input_flat.numel()

    BLOCK_SIZE = 1024
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    index_boolean_mask_kernel[grid](
        input_flat,
        mask_flat,
        output,
        write_indices,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def take_triton(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Take elements from flattened input.

    Args:
        input: Input tensor (any shape)
        indices: Flat indices

    Returns:
        Gathered elements
    """
    input_flat = input.flatten()
    indices_flat = indices.flatten()

    n_elements = indices_flat.numel()
    input_size = input_flat.numel()

    output = torch.empty_like(indices_flat, dtype=input.dtype)

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    index_1d_kernel[grid](
        input_flat,
        indices_flat,
        output,
        n_elements,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(indices.shape)


@triton.jit
def index_select_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    n_output_elements,
    n_indices,
    input_dim_size,
    other_dims_before,  # Product of dims before indexed dim
    other_dims_after,  # Product of dims after indexed dim
    input_stride_before,
    input_stride_indexed,
    input_stride_after,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Index select along a specific dimension.

    Similar to torch.index_select but implemented in Triton.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_output_elements

    # Decompose output offset into (before, index_idx, after)
    after_idx = offsets % other_dims_after
    temp = offsets // other_dims_after
    index_idx = temp - (temp // n_indices) * n_indices
    before_idx = temp // n_indices

    # Load the selected index
    selected_idx = tl.load(indices_ptr + index_idx, mask=mask, other=0)

    # Handle negative indices
    selected_idx = tl.where(
        selected_idx < 0, selected_idx + input_dim_size, selected_idx
    )
    selected_idx = tl.minimum(tl.maximum(selected_idx, 0), input_dim_size - 1)

    # Compute input offset
    input_offset = (
        before_idx * input_stride_before
        + selected_idx * input_stride_indexed
        + after_idx * input_stride_after
    )

    # Load and store
    value = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def index_select_triton(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """
    Select indices along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to select along
        index: 1D tensor of indices

    Returns:
        Tensor with selected slices
    """
    assert index.dim() == 1, "Index must be 1D"

    if dim < 0:
        dim += input.dim()

    # Compute output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.numel()

    # Compute dimension products
    other_dims_before = 1
    for i in range(dim):
        other_dims_before *= input.size(i)

    other_dims_after = 1
    for i in range(dim + 1, input.dim()):
        other_dims_after *= input.size(i)

    n_indices = index.numel()
    n_output_elements = torch.prod(torch.tensor(output_shape)).item()
    input_dim_size = input.size(dim)

    # Compute strides
    input_stride_before = input.stride(dim - 1) if dim > 0 else input.numel()
    input_stride_indexed = input.stride(dim)
    input_stride_after = input.stride(dim + 1) if dim < input.dim() - 1 else 1

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    input = input.contiguous()
    index = index.contiguous()

    BLOCK_SIZE = triton.next_power_of_2(n_output_elements)
    # grid = lambda meta: (triton.cdiv(n_output_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    index_select_kernel[grid](
        input,
        index,
        output,
        n_output_elements,
        n_indices,
        input_dim_size,
        other_dims_before,
        other_dims_after,
        input_stride_before,
        input_stride_indexed,
        input_stride_after,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_index_1d():
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32)

    output = index_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


@dsp
def test_index_negative_idx():
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = torch.tensor([0, -1, -2, 2], dtype=torch.int32)

    output = index_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


@dsp
def test_index_2d_single():
    x = torch.randn(5, 4)
    indices = torch.tensor([0, 2, 4], dtype=torch.int32)

    output = index_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


@dsp
def test_index_2d_double():
    x = torch.randn(5, 6)
    indices_row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    indices_col = torch.tensor([0, 2, 4, 5], dtype=torch.int32)

    output = index_triton(x, (indices_row, indices_col))
    reference = x[indices_row, indices_col]

    assert torch.allclose(output, reference)


@dsp
def test_index_mask():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True])

    output = boolean_index_triton(x, mask)
    reference = x[mask]

    assert torch.allclose(output, reference)


@dsp
def test_index_slice():
    x = torch.arange(10, dtype=torch.float32)

    output = index_triton(x, slice(2, 8, 2))
    reference = x[2:8:2]

    assert torch.allclose(output, reference)


@dsp
def test_index_select():
    x = torch.randn(5, 10)
    indices = torch.tensor([0, 2, 4, 7], dtype=torch.int32)

    output = index_select_triton(x, 1, indices)
    reference = torch.index_select(x, 1, indices)

    assert torch.allclose(output, reference)


@dsp
def test_index_take():
    x = torch.randn(20)
    indices = torch.tensor([5, 10, 15, 0, 19], dtype=torch.int32)

    output = take_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


@dsp
def test_index_broadcasting():
    x = torch.randn(5, 6)
    idx_row = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    idx_col = torch.tensor([[0, 2], [4, 5]], dtype=torch.int32)

    output = index_triton(x, (idx_row, idx_col))
    reference = x[idx_row, idx_col]

    assert torch.allclose(output, reference)


@dsp
def test_index_large():
    x = torch.randn(1000)
    indices = torch.randint(0, 1000, (500,), dtype=torch.int32)

    output = index_triton(x, indices)
    reference = x[indices]

    assert torch.allclose(output, reference)


def benchmark_index():
    """Benchmark indexing operations."""
    import time

    print("\nBenchmarking index operations...\n")

    # Benchmark 1D indexing
    print("1D Index Benchmark:")
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        input = torch.randn(size)
        indices = torch.randint(0, size, (size // 2,))

        # Warmup
        for _ in range(10):
            _ = input[indices]
            _ = index_triton(input, indices)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = input[indices]
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = index_triton(input, indices)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark 2D indexing
    print("\n2D Index Benchmark:")
    shapes = [(100, 100), (512, 512), (1024, 1024)]

    for M, N in shapes:
        input = torch.randn(M, N)
        indices = torch.randint(0, M, (M // 2,))

        # Warmup
        for _ in range(10):
            _ = input[indices]
            _ = index_triton(input, indices)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = input[indices]
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = index_triton(input, indices)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Shape {M}x{N}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark_index()
