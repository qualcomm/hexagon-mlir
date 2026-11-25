# ===- test_dsp_index_put.py ------------------------------------------------===
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

from .. import dsp, fixme

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def index_put_1d_kernel(
    input_ptr,  # Pointer to input/output tensor
    indices_ptr,  # Pointer to indices (list of 1D tensors)
    values_ptr,  # Pointer to values to put
    n_elements,  # Number of elements to put
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE: tl.constexpr,  # Whether to accumulate (add) or assign
):
    """
    1D index_put kernel.

    input[indices] = values (if not accumulate)
    input[indices] += values (if accumulate)

    Args:
        input_ptr: Input/output tensor (modified in-place)
        indices_ptr: Indices where to put values
        values_ptr: Values to put
        n_elements: Number of elements to put
        accumulate: If True, add values; if False, assign
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices and values
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)

    # Put values at indices
    if ACCUMULATE:
        # Use atomic add for accumulation
        tl.atomic_add(input_ptr + indices, values, mask=mask)
    else:
        # Direct assignment (may have race conditions if duplicate indices)
        tl.store(input_ptr + indices, values, mask=mask)


@triton.jit
def index_put_2d_single_index_kernel(
    input_ptr,
    indices_0_ptr,  # Indices for dimension 0
    values_ptr,
    n_elements,  # Number of index elements
    stride_0,  # Stride for dimension 0
    stride_1,  # Stride for dimension 1
    dim_1,  # Size of dimension 1
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """
    2D index_put with indexing on first dimension only.

    input[indices_0, :] = values

    This is the common pattern: input[idx] = values for 2D tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Each thread handles one (index, col) pair
    flat_offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = flat_offset < (n_elements * dim_1)

    # Decompose into (index_idx, col)
    col = flat_offset % dim_1
    idx_num = flat_offset // dim_1

    # Load the row index
    row_idx = tl.load(indices_0_ptr + idx_num, mask=mask, other=0)

    # Compute input offset
    input_offset = row_idx * stride_0 + col * stride_1

    # Load value
    value = tl.load(values_ptr + flat_offset, mask=mask, other=0.0)

    # Put value
    if ACCUMULATE:
        tl.atomic_add(input_ptr + input_offset, value, mask=mask)
    else:
        tl.store(input_ptr + input_offset, value, mask=mask)


@triton.jit
def index_put_2d_dual_index_kernel(
    input_ptr,
    indices_0_ptr,  # Indices for dimension 0
    indices_1_ptr,  # Indices for dimension 1
    values_ptr,
    n_elements,
    stride_0,
    stride_1,
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """
    2D index_put with indexing on both dimensions.

    input[indices_0, indices_1] = values

    Both index tensors should have the same length.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices for both dimensions
    idx_0 = tl.load(indices_0_ptr + offsets, mask=mask, other=0)
    idx_1 = tl.load(indices_1_ptr + offsets, mask=mask, other=0)

    # Compute input offset
    input_offset = idx_0 * stride_0 + idx_1 * stride_1

    # Load value
    value = tl.load(values_ptr + offsets, mask=mask, other=0.0)

    # Put value
    if ACCUMULATE:
        tl.atomic_add(input_ptr + input_offset, value, mask=mask)
    else:
        tl.store(input_ptr + input_offset, value, mask=mask)


@triton.jit
def index_put_nd_kernel(
    input_ptr,
    values_ptr,
    n_elements,  # Number of elements to put
    n_dims,  # Number of dimensions being indexed
    indices_ptrs,  # Array of pointers to index tensors
    strides_ptr,  # Pointer to stride array
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """
    N-dimensional index_put kernel.

    Handles arbitrary number of index tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # For each element, compute the flat offset into input
    # This requires loading multiple index values and combining with strides

    # Initialize offset accumulator
    input_offset = tl.zeros_like(offsets)

    # Load and combine indices for each dimension
    # Note: This is simplified - in practice, need to handle pointer array
    for dim in range(n_dims):
        # Load index for this dimension
        # idx_ptr = tl.load(indices_ptrs + dim) # Not supported in Triton
        # idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

        # stride = tl.load(strides_ptr + dim)
        # input_offset += idx * stride
        pass

    # Load value
    value = tl.load(values_ptr + offsets, mask=mask, other=0.0)

    # Put value
    if ACCUMULATE:
        tl.atomic_add(input_ptr + input_offset, value, mask=mask)
    else:
        tl.store(input_ptr + input_offset, value, mask=mask)


@triton.jit
def index_put_masked_kernel(
    input_ptr,
    mask_ptr,  # Boolean mask
    values_ptr,
    n_elements,  # Total elements in input
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """
    Index put using a boolean mask.

    input[mask] = values

    Values should have size equal to mask.sum().
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load mask
    mask_bool = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

    # Count how many True values before this offset
    # This is complex - needs prefix sum
    # Simplified: we use atomic operations and let duplicates handle themselves

    # For now, we handle this by having the caller convert mask to indices
    # and use the regular index_put
    pass


@triton.jit
def index_add_kernel(
    input_ptr,
    dim,
    indices_ptr,
    source_ptr,
    n_elements,  # Number of elements in indices/source
    input_stride_0,
    input_stride_1,
    source_stride_0,
    source_stride_1,
    input_dim_0,
    input_dim_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Index add operation: input.index_add_(dim, indices, source)

    For each i: input[..., indices[i], ...] += source[..., i, ...]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # This is complex for general N-D
    # Simplified for 2D case with dim=0
    if dim == 0:
        # Decompose offset into (idx_num, col)
        n_total = n_elements * input_dim_1
        mask = offsets < n_total

        col = offsets % input_dim_1
        idx_num = offsets // input_dim_1

        # Load the index
        idx = tl.load(indices_ptr + idx_num, mask=mask, other=0)

        # Compute offsets
        input_offset = idx * input_stride_0 + col * input_stride_1
        source_offset = idx_num * source_stride_0 + col * source_stride_1

        # Load source value
        value = tl.load(source_ptr + source_offset, mask=mask, other=0.0)

        # Add to input
        tl.atomic_add(input_ptr + input_offset, value, mask=mask)


def index_put_triton(
    input: torch.Tensor, indices: tuple, values: torch.Tensor, accumulate: bool = False
) -> torch.Tensor:
    """
    PyTorch wrapper for Triton index_put operation.

    Puts values into input at locations specified by indices.

    Args:
        input: Input tensor (N-D)
        indices: Tuple of index tensors (can include None for full slices)
        values: Values to put (should broadcast to indexed shape)
        accumulate: If True, add values; if False, assign

    Returns:
        Modified input tensor

    Examples:
        >>> input = torch.zeros(5, 5)
        >>> indices = (torch.tensor([0, 2, 4]), torch.tensor([1, 3, 0]))
        >>> values = torch.tensor([1.0, 2.0, 3.0])
        >>> index_put_triton(input, indices, values)
    """

    # Filter out None indices
    non_none_indices = [idx for idx in indices if idx is not None]

    if len(non_none_indices) == 0:
        # No indexing, just assign all values
        if accumulate:
            input += values
        else:
            input.copy_(values)
        return input

    # Make input contiguous
    input = input.contiguous()

    # Handle different cases based on number of indices
    if len(non_none_indices) == 1 and input.dim() == 1:
        # 1D case: input[indices] = values
        indices_0 = non_none_indices[0].contiguous()
        values_flat = values.flatten().contiguous()
        n_elements = indices_0.numel()

        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        index_put_1d_kernel[grid](
            input,
            indices_0,
            values_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            ACCUMULATE=accumulate,
        )

    elif len(non_none_indices) == 1 and input.dim() == 2:
        # 2D case with single index: input[indices, :] = values
        indices_0 = non_none_indices[0].contiguous()
        values_flat = values.flatten().contiguous()
        n_indices = indices_0.numel()

        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_indices * input.size(1), BLOCK_SIZE),)

        index_put_2d_single_index_kernel[grid](
            input,
            indices_0,
            values_flat,
            n_indices,
            input.stride(0),
            input.stride(1),
            input.size(1),
            BLOCK_SIZE=BLOCK_SIZE,
            ACCUMULATE=accumulate,
        )

    elif len(non_none_indices) == 2 and input.dim() == 2:
        # 2D case with dual indices: input[indices_0, indices_1] = values
        indices_0 = non_none_indices[0].contiguous()
        indices_1 = non_none_indices[1].contiguous()

        # Broadcast indices to same shape
        indices_0_bc, indices_1_bc = torch.broadcast_tensors(indices_0, indices_1)
        indices_0_flat = indices_0_bc.flatten()
        indices_1_flat = indices_1_bc.flatten()

        values_flat = values.flatten().contiguous()
        n_elements = indices_0_flat.numel()

        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        index_put_2d_dual_index_kernel[grid](
            input,
            indices_0_flat,
            indices_1_flat,
            values_flat,
            n_elements,
            input.stride(0),
            input.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            ACCUMULATE=accumulate,
        )

    else:
        # Fall back to PyTorch for complex cases
        if accumulate:
            # Use index_put with accumulate
            input.index_put_(indices, values, accumulate=True)
        else:
            input.index_put_(indices, values, accumulate=False)

    return input


def index_put_triton_(
    input: torch.Tensor, indices: tuple, values: torch.Tensor, accumulate: bool = False
) -> torch.Tensor:
    """
    In-place version of index_put.
    """
    return index_put_triton(input, indices, values, accumulate)


def masked_fill_triton_(
    input: torch.Tensor, mask: torch.Tensor, value: float
) -> torch.Tensor:
    """
    Fill elements of input with value where mask is True.

    Equivalent to input[mask] = value
    """
    assert input.shape == mask.shape, "Input and mask must have same shape"
    assert mask.dtype == torch.bool, "Mask must be boolean"

    # Convert mask to indices
    indices_flat = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1)

    if indices_flat.numel() == 0:
        return input

    # Create values tensor
    values = torch.full(
        (indices_flat.numel(),), value, dtype=input.dtype, device=input.device
    )

    # Flatten input temporarily
    input_flat = input.flatten()

    BLOCK_SIZE = 256
    grid = (triton.cdiv(indices_flat.numel(), BLOCK_SIZE),)

    index_put_1d_kernel[grid](
        input_flat,
        indices_flat,
        values,
        indices_flat.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        ACCUMULATE=False,
    )

    return input


def masked_scatter_triton_(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    Scatter values from source into input where mask is True.

    Equivalent to input[mask] = source
    """
    assert input.shape == mask.shape
    assert mask.dtype == torch.bool

    # Get indices where mask is True
    indices_flat = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1)
    n_true = indices_flat.numel()

    if n_true == 0:
        return input

    # Get values from source
    source_flat = source.flatten()
    assert source_flat.numel() >= n_true, "Source has insufficient elements"
    values = source_flat[:n_true]

    # Flatten input
    input_flat = input.flatten()

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_true, BLOCK_SIZE),)

    index_put_1d_kernel[grid](
        input_flat,
        indices_flat,
        values,
        n_true,
        BLOCK_SIZE=BLOCK_SIZE,
        ACCUMULATE=False,
    )

    return input


@triton.jit
def index_fill_kernel(
    input_ptr,
    indices_ptr,
    value,
    n_indices,
    stride_0,
    stride_1,
    dim_1,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
):
    """
    Fill indexed slices with a scalar value.

    input[indices, :] = value (if dim=0)
    input[:, indices] = value (if dim=1)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    if DIM == 0:
        # Fill rows
        n_total = n_indices * dim_1
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_total

        col = offsets % dim_1
        idx_num = offsets // dim_1

        row_idx = tl.load(indices_ptr + idx_num, mask=mask, other=0)
        input_offset = row_idx * stride_0 + col * stride_1

        tl.store(input_ptr + input_offset, value, mask=mask)
    else:
        # Fill columns
        n_total = n_indices * dim_1
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_total

        row = offsets % dim_1
        idx_num = offsets // dim_1

        col_idx = tl.load(indices_ptr + idx_num, mask=mask, other=0)
        input_offset = row * stride_0 + col_idx * stride_1

        tl.store(input_ptr + input_offset, value, mask=mask)


def index_fill_triton_(
    input: torch.Tensor, dim: int, index: torch.Tensor, value: float
) -> torch.Tensor:
    """
    Fill indexed slices with a scalar value.

    Args:
        input: Input tensor (2D)
        dim: Dimension to index along
        index: 1D tensor of indices
        value: Scalar value to fill

    Returns:
        Modified input tensor
    """
    assert input.dim() == 2, "This implementation supports 2D tensors"
    assert index.dim() == 1, "Index must be 1D"

    input = input.contiguous()
    index = index.contiguous()

    n_indices = index.numel()
    other_dim = input.size(1 - dim)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_indices * other_dim, BLOCK_SIZE),)

    index_fill_kernel[grid](
        input,
        index,
        value,
        n_indices,
        input.stride(0),
        input.stride(1),
        other_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM=dim,
    )

    return input


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_index_put_1d():
    x = torch.zeros(10)
    indices = (torch.tensor([0, 2, 5, 7]),)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    output = x.clone()
    reference = x.clone()

    index_put_triton_(output, indices, values)
    reference.index_put_(indices, values)

    assert torch.allclose(output, reference)


@fixme
@dsp
def test_index_put_1d_acc():
    x = torch.ones(10)
    indices = (torch.tensor([0, 0, 1, 1]),)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    output = x.clone()
    reference = x.clone()

    index_put_triton_(output, indices, values, accumulate=True)
    reference.index_put_(indices, values, accumulate=True)

    assert torch.allclose(output, reference)


@dsp
def test_index_put_2d_singleidx():
    x = torch.zeros(5, 4)
    indices = (torch.tensor([0, 2, 4]),)
    values = torch.ones(3, 4)

    output = x.clone()
    reference = x.clone()

    index_put_triton_(output, indices, values)
    reference.index_put_(indices, values)

    assert torch.allclose(output, reference)


@dsp
def test_index_put_2d_dualidx():
    x = torch.zeros(5, 5)
    indices = (
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([0, 1, 2, 3, 4]),
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    output = x.clone()
    reference = x.clone()

    index_put_triton_(output, indices, values)
    reference.index_put_(indices, values)

    assert torch.allclose(output, reference)


@dsp
def test_masked_fill():
    x = torch.zeros(10)
    mask = torch.tensor(
        [True, False, True, False, True, False, True, False, True, False]
    )
    value = 99.0

    output = x.clone()
    reference = x.clone()

    masked_fill_triton_(output, mask, value)
    reference.masked_fill_(mask, value)

    assert torch.allclose(output, reference)


@dsp
def test_masked_scatter():
    x = torch.zeros(10)
    mask_scatter = torch.tensor(
        [True, False, True, False, True, False, True, False, False, False]
    )
    source_scatter = torch.tensor([1.0, 2.0, 3.0, 4.0])

    output = x.clone()
    reference = x.clone()

    masked_scatter_triton_(output, mask_scatter, source_scatter)
    reference.masked_scatter_(mask_scatter, source_scatter)

    assert torch.allclose(output, reference)


@dsp
def test_index_fill():
    x = torch.zeros(5, 4)
    indices = torch.tensor([0, 2, 4])
    value_fill = 7.0

    output = x.clone()
    reference = x.clone()

    index_fill_triton_(output, 0, indices, value_fill)
    reference.index_fill_(0, indices, value_fill)

    assert torch.allclose(output, reference)


@fixme
@dsp
def test_index_put_large():
    x = torch.zeros(1000)
    indices = (torch.randint(0, 1000, (500,)),)
    values = torch.randn(500)

    output = x.clone()
    reference = x.clone()

    index_put_triton_(output, indices, values, accumulate=True)
    reference.index_put_(indices, values, accumulate=True)

    assert torch.allclose(output, reference)


def benchmark_index_put():
    """Benchmark index_put operations."""
    import time

    print("\nBenchmarking index_put operations...\n")

    # Benchmark 1D index_put
    print("1D Index Put Benchmark:")
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        input = torch.zeros(size)
        indices = (torch.randint(0, size, (size // 2,)),)
        values = torch.randn(size // 2)

        # Warmup
        for _ in range(10):
            inp_copy = input.clone()
            inp_copy.index_put_(indices, values)

            inp_copy2 = input.clone()
            index_put_triton_(inp_copy2, indices, values)


        n_iters = 100

        # PyTorch
        start = time.time()
        for _ in range(n_iters):
            inp_copy = input.clone()
            inp_copy.index_put_(indices, values)
        time_torch = (time.time() - start) / n_iters * 1000

        # Triton
        start = time.time()
        for _ in range(n_iters):
            inp_copy = input.clone()
            index_put_triton_(inp_copy, indices, values)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark with accumulate (gradient accumulation scenario)
    print("\n1D Index Put with Accumulate:")
    for size in sizes:
        input = torch.zeros(size)
        # Create duplicate indices for realistic accumulation
        indices = (torch.randint(0, size // 10, (size,)),)
        values = torch.randn(size)

        # Warmup
        for _ in range(10):
            inp_copy = input.clone()
            inp_copy.index_put_(indices, values, accumulate=True)

            inp_copy2 = input.clone()
            index_put_triton_(inp_copy2, indices, values, accumulate=True)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            inp_copy = input.clone()
            inp_copy.index_put_(indices, values, accumulate=True)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            inp_copy = input.clone()
            index_put_triton_(inp_copy, indices, values, accumulate=True)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark_index_put()
