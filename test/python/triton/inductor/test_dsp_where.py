# ===- test_dsp_where.py ----------------------------------------------------===
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
def where_kernel(
    condition_ptr,  # Pointer to condition tensor (boolean)
    x_ptr,  # Pointer to x tensor (values when True)
    y_ptr,  # Pointer to y tensor (values when False)
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Basic where kernel: output = condition ? x : y

    Element-wise selection based on condition.

    Args:
        condition_ptr: Boolean condition tensor
        x_ptr: Values to use where condition is True
        y_ptr: Values to use where condition is False
        output_ptr: Output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load condition, x, and y
    condition = tl.load(condition_ptr + offsets, mask=mask, other=False)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Select based on condition
    output = tl.where(condition, x, y)

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def where_scalar_x_kernel(
    condition_ptr,
    x_scalar,  # Scalar value for True case
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Where kernel with scalar x: output = condition ? x_scalar : y
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask, other=False)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Broadcast scalar x
    x_vec = tl.full([BLOCK_SIZE], x_scalar, dtype=y.dtype)

    output = tl.where(condition, x_vec, y)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def where_scalar_y_kernel(
    condition_ptr,
    x_ptr,
    y_scalar,  # Scalar value for False case
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Where kernel with scalar y: output = condition ? x : y_scalar
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask, other=False)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Broadcast scalar y
    y_vec = tl.full([BLOCK_SIZE], y_scalar, dtype=x.dtype)

    output = tl.where(condition, x, y_vec)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def where_both_scalars_kernel(
    condition_ptr,
    x_scalar,
    y_scalar,
    output_ptr,
    n_elements,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Where kernel with both scalars: output = condition ? x_scalar : y_scalar
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask, other=False)

    # Create scalar vectors
    x_vec = tl.full([BLOCK_SIZE], x_scalar, dtype=output_dtype)
    y_vec = tl.full([BLOCK_SIZE], y_scalar, dtype=output_dtype)

    output = tl.where(condition, x_vec, y_vec)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def where_2d_kernel(
    condition_ptr,
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    condition_stride_0,
    condition_stride_1,
    x_stride_0,
    x_stride_1,
    y_stride_0,
    y_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D where kernel with better memory coalescing.
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

    # Compute offsets
    condition_offsets = offs_m * condition_stride_0 + offs_n * condition_stride_1
    x_offsets = offs_m * x_stride_0 + offs_n * x_stride_1
    y_offsets = offs_m * y_stride_0 + offs_n * y_stride_1
    output_offsets = offs_m * output_stride_0 + offs_n * output_stride_1

    # Load data
    condition = tl.load(condition_ptr + condition_offsets, mask=mask, other=False)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)

    # Select
    output = tl.where(condition, x, y)

    # Store
    tl.store(output_ptr + output_offsets, output, mask=mask)


@triton.jit
def where_nonzero_kernel(
    condition_ptr,
    output_ptr,  # Output indices (flattened)
    count_ptr,  # Pointer to count of True elements
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Returns indices where condition is True (equivalent to torch.nonzero).

    This is a simplified version - full implementation needs:
    1. Count pass to determine output size
    2. Compaction pass to write indices
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load condition
    condition = tl.load(condition_ptr + offsets, mask=mask, other=False)

    # Count True values (this would need atomic add for correct counting)
    true_count = tl.sum(condition.to(tl.int32))

    # This is simplified - real implementation would need:
    # - Prefix sum to determine write positions
    # - Compaction to write indices
    # For now, we mark this as a placeholder


@triton.jit
def where_broadcast_kernel(
    condition_ptr,
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    # Broadcasting information
    condition_size,
    x_size,
    y_size,
    condition_stride,
    x_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Where kernel with broadcasting support.
    Handles cases where inputs have different shapes.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices with broadcasting
    # If size is 1, index is always 0 (broadcast)
    # Otherwise, use modulo for cycling
    condition_idx = tl.where(condition_size == 1, 0, offsets % condition_size)
    x_idx = tl.where(x_size == 1, 0, offsets % x_size)
    y_idx = tl.where(y_size == 1, 0, offsets % y_size)

    # Load with broadcasting
    condition = tl.load(
        condition_ptr + condition_idx * condition_stride, mask=mask, other=False
    )
    x = tl.load(x_ptr + x_idx * x_stride, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_idx * y_stride, mask=mask, other=0.0)

    # Select
    output = tl.where(condition, x, y)

    # Store
    tl.store(output_ptr + offsets, output, mask=mask)


def where_triton(
    condition: torch.Tensor, x: torch.Tensor | float, y: torch.Tensor | float
) -> torch.Tensor:
    """
    PyTorch wrapper for Triton where operation.

    Returns elements from x where condition is True, otherwise from y.

    Args:
        condition: Boolean condition tensor
        x: Values when condition is True (tensor or scalar)
        y: Values when condition is False (tensor or scalar)

    Returns:
        Output tensor with selected values

    Examples:
        >>> condition = torch.tensor([True, False, True, False])
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> y = torch.tensor([5, 6, 7, 8])
        >>> where_triton(condition, x, y)
        tensor([1, 6, 3, 8])
    """
    assert condition.dtype == torch.bool, "Condition must be boolean"

    # Determine if x and y are scalars or tensors
    x_is_scalar = isinstance(x, (int, float))
    y_is_scalar = isinstance(y, (int, float))

    # Ensure condition is on CUDA
    # if not condition.is_cuda:
    #    condition = condition.cuda()

    # Handle different combinations of scalar/tensor inputs
    if x_is_scalar and y_is_scalar:
        # Both scalars
        n_elements = condition.numel()

        # Determine output dtype
        if isinstance(x, int) and isinstance(y, int):
            output_dtype = torch.int64
            triton_dtype = tl.int64
        else:
            output_dtype = torch.float32
            triton_dtype = tl.float32

        output = torch.empty_like(condition, dtype=output_dtype)
        condition_flat = condition.flatten()
        output_flat = output.flatten()

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        where_both_scalars_kernel[grid](
            condition_flat,
            float(x),
            float(y),
            output_flat,
            n_elements,
            triton_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output.reshape(condition.shape)

    elif x_is_scalar:
        # x is scalar, y is tensor
        # if not y.is_cuda:
        #    y = y.cuda()

        # Broadcast shapes
        output_shape = torch.broadcast_shapes(condition.shape, y.shape)
        condition_bc = condition.broadcast_to(output_shape)
        y_bc = y.broadcast_to(output_shape)

        n_elements = condition_bc.numel()
        output = torch.empty_like(y_bc)

        condition_flat = condition_bc.flatten()
        y_flat = y_bc.flatten()
        output_flat = output.flatten()

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        where_scalar_x_kernel[grid](
            condition_flat,
            float(x),
            y_flat,
            output_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output.reshape(output_shape)

    elif y_is_scalar:
        # y is scalar, x is tensor
        # if not x.is_cuda:
        #    x = x.cuda()

        # Broadcast shapes
        output_shape = torch.broadcast_shapes(condition.shape, x.shape)
        condition_bc = condition.broadcast_to(output_shape)
        x_bc = x.broadcast_to(output_shape)

        n_elements = condition_bc.numel()
        output = torch.empty_like(x_bc)

        condition_flat = condition_bc.flatten()
        x_flat = x_bc.flatten()
        output_flat = output.flatten()

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        where_scalar_y_kernel[grid](
            condition_flat,
            x_flat,
            float(y),
            output_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output.reshape(output_shape)

    else:
        # Both are tensors
        # if not x.is_cuda:
        #    x = x.cuda()
        # if not y.is_cuda:
        #    y = y.cuda()

        # Broadcast shapes
        output_shape = torch.broadcast_shapes(condition.shape, x.shape, y.shape)
        condition_bc = condition.broadcast_to(output_shape).contiguous()
        x_bc = x.broadcast_to(output_shape).contiguous()
        y_bc = y.broadcast_to(output_shape).contiguous()

        output = torch.empty_like(x_bc)

        # Use 2D kernel for better performance if 2D
        if len(output_shape) == 2:
            M, N = output_shape

            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 32

            grid = (
                triton.cdiv(M, BLOCK_SIZE_M),
                triton.cdiv(N, BLOCK_SIZE_N),
            )
            # grid = (1,1,)

            where_2d_kernel[grid](
                condition_bc,
                x_bc,
                y_bc,
                output,
                M,
                N,
                condition_bc.stride(0),
                condition_bc.stride(1),
                x_bc.stride(0),
                x_bc.stride(1),
                y_bc.stride(0),
                y_bc.stride(1),
                output.stride(0),
                output.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        else:
            # Flatten for 1D kernel
            n_elements = output.numel()
            condition_flat = condition_bc.flatten()
            x_flat = x_bc.flatten()
            y_flat = y_bc.flatten()
            output_flat = output.flatten()

            BLOCK_SIZE = triton.next_power_of_2(n_elements)
            # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            grid = (1,)

            where_kernel[grid](
                condition_flat,
                x_flat,
                y_flat,
                output_flat,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return output.reshape(output_shape)


def nonzero_triton(condition: torch.Tensor) -> torch.Tensor:
    """
    Returns indices where condition is True.

    This is a simplified wrapper that uses PyTorch's nonzero
    since efficient GPU compaction is complex in Triton.

    For a full Triton implementation, you'd need:
    1. Count pass to determine output size
    2. Prefix sum for write positions
    3. Compaction pass to write indices
    """
    # For now, fall back to PyTorch
    return torch.nonzero(condition)


@triton.jit
def masked_select_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    write_indices_ptr,  # Prefix sum of mask (where to write each element)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Select elements from input where mask is True.
    Requires precomputed prefix sum for write positions.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements

    # Load mask and input
    mask = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)
    input_val = tl.load(input_ptr + offsets, mask=mask_valid, other=0.0)

    # Load write index (from prefix sum)
    write_idx = tl.load(write_indices_ptr + offsets, mask=mask_valid, other=0).to(
        tl.int32
    )

    # Write to output if mask is True
    tl.store(output_ptr + write_idx, input_val, mask=mask_valid & mask)


def masked_select_triton(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Select elements from input where mask is True.

    Args:
        input: Input tensor
        mask: Boolean mask (same shape as input)

    Returns:
        1D tensor with selected elements
    """
    assert input.shape == mask.shape
    assert mask.dtype == torch.bool

    # Flatten tensors
    input_flat = input.flatten()
    mask_flat = mask.flatten()

    # Compute prefix sum to determine write positions
    mask_int = mask_flat.to(torch.int32)
    write_indices = (torch.cumsum(mask_int, dim=0) - mask_int).to(torch.int32)

    # Count total True values
    n_true = mask_flat.sum().item()

    if n_true == 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    # Allocate output
    output = torch.empty(n_true, dtype=input.dtype, device=input.device)

    n_elements = input_flat.numel()

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (1,)

    masked_select_kernel[grid](
        input_flat,
        mask_flat,
        output,
        write_indices,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_where_1d():
    condition = torch.tensor([True, False, True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    output = where_triton(condition, x, y)
    reference = torch.where(condition, x, y)

    assert torch.allclose(output, reference)


@dsp
def test_where_scalar_x():
    condition = torch.tensor([True, False, True, False, True])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    output = where_triton(condition, 99.0, y)
    reference = torch.where(condition, 99.0, y)

    assert torch.allclose(output, reference)


@dsp
def test_where_scalar_y():
    condition = torch.tensor([True, False, True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    output = where_triton(condition, x, -1.0)
    reference = torch.where(condition, x, -1.0)

    assert torch.allclose(output, reference)


@dsp
def test_where_scalar_xy():
    condition = torch.tensor([True, False, True, False, True])

    output = where_triton(condition, 1.0, 0.0)
    reference = torch.where(condition, 1.0, 0.0)

    assert torch.allclose(output, reference)


@dsp
def test_where_2d():
    condition = torch.rand(4, 5) > 0.5
    x = torch.randn(4, 5)
    y = torch.randn(4, 5)

    output = where_triton(condition, x, y)
    reference = torch.where(condition, x, y)

    assert torch.allclose(output, reference)


@dsp
def test_where_broadcasting():
    condition_ = torch.tensor([True, False, True])
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    output = where_triton(condition_.unsqueeze(1), x, y)
    reference = torch.where(condition_.unsqueeze(1), x, y)

    assert torch.allclose(output, reference)


@dsp
def test_where_masked():
    mask = torch.tensor([True, False, True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    output = masked_select_triton(x, mask)
    reference = x[mask]

    assert torch.allclose(output, reference)


@dsp
def test_where_large():
    condition = torch.rand(1000, 1000) > 0.5
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    reference = torch.where(condition, x, y)
    output = where_triton(condition, x, y)

    assert torch.allclose(output, reference)


@dsp
@parametrize_dtypes(torch.float16, torch.float32, torch.int32)
def test_where_dtypes(dtype):
    if dtype == torch.int32:
        x = torch.randint(0, 100, (100,), dtype=dtype)
        y = torch.randint(0, 100, (100,), dtype=dtype)
    else:
        x = torch.randn(100, dtype=dtype)
        y = torch.randn(100, dtype=dtype)

    condition = torch.rand(100) > 0.5

    output = where_triton(condition, x, y)
    reference = torch.where(condition, x, y)

    assert torch.allclose(reference.float(), output.float(), rtol=1e-3)


def benchmark_where():
    """Benchmark where operations."""
    import time

    print("\nBenchmarking where operations...\n")

    # Benchmark 1D where
    print("1D Where Benchmark:")
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        condition = torch.rand(size) > 0.5
        x = torch.randn(size)
        y = torch.randn(size)

        # Warmup
        for _ in range(10):
            _ = torch.where(condition, x, y)
            _ = where_triton(condition, x, y)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = torch.where(condition, x, y)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = where_triton(condition, x, y)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark 2D where
    print("\n2D Where Benchmark:")
    shapes = [(100, 100), (512, 512), (1024, 1024), (2048, 2048)]

    for M, N in shapes:
        condition = torch.rand(M, N) > 0.5
        x = torch.randn(M, N)
        y = torch.randn(M, N)

        # Warmup
        for _ in range(10):
            _ = torch.where(condition, x, y)
            _ = where_triton(condition, x, y)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = torch.where(condition, x, y)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = where_triton(condition, x, y)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Shape {M}x{N}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark with scalars
    print("\nWhere with Scalars Benchmark:")
    for size in [100000, 1000000]:
        condition = torch.rand(size) > 0.5
        x = torch.randn(size)

        # Warmup
        for _ in range(10):
            _ = torch.where(condition, x, 0.0)
            _ = where_triton(condition, x, 0.0)


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            _ = torch.where(condition, x, 0.0)
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            _ = where_triton(condition, x, 0.0)
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7} (scalar y): PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark_where()
