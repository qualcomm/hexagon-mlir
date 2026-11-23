# ===- test_dsp_scatter.py --------------------------------------------------===
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

from .. import dsp, fixme, parametrize_dims, parametrize_reduces

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


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


@triton.jit
def scatter_add_kernel(
    input_ptr,
    indices_ptr,
    src_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter with add reduction - atomically adds values at scatter locations.

    output[indices[i]] += src[i]
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

    # Atomic add to handle multiple writes to same location
    tl.atomic_add(output_ptr + indices, src_values, mask=scatter_mask)


@triton.jit
def scatter_mul_kernel(
    input_ptr,
    indices_ptr,
    src_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter with multiply reduction.

    output[indices[i]] *= src[i]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    src_values = tl.load(src_ptr + offsets, mask=mask, other=1.0)

    valid_indices = (indices >= 0) & (indices < output_size)
    scatter_mask = mask & valid_indices

    # Atomic multiply
    # Note: Triton may not have atomic_mul, so we simulate with CAS loop
    # For simplicity, using non-atomic version (races possible)

    # Load current value, multiply, store back
    # This is NOT atomic and will have races
    for i in range(BLOCK_SIZE):
        if offsets[i] < n_elements and scatter_mask[i]:
            idx = indices[i]
            current = tl.load(output_ptr + idx)
            new_val = current * src_values[i]
            tl.store(output_ptr + idx, new_val)


@triton.jit
def scatter_max_kernel(
    input_ptr,
    indices_ptr,
    src_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter with max reduction.

    output[indices[i]] = max(output[indices[i]], src[i])
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    src_values = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    valid_indices = (indices >= 0) & (indices < output_size)
    scatter_mask = mask & valid_indices

    # Atomic max
    tl.atomic_max(output_ptr + indices, src_values, mask=scatter_mask)


@triton.jit
def scatter_min_kernel(
    input_ptr,
    indices_ptr,
    src_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter with min reduction.

    output[indices[i]] = min(output[indices[i]], src[i])
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    src_values = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    valid_indices = (indices >= 0) & (indices < output_size)
    scatter_mask = mask & valid_indices

    # Atomic min
    tl.atomic_min(output_ptr + indices, src_values, mask=scatter_mask)


@triton.jit
def scatter_2d_kernel(
    input_ptr,
    indices_ptr,
    src_ptr,
    output_ptr,
    M,
    N,  # Dimensions
    input_stride_0,
    input_stride_1,
    indices_stride_0,
    indices_stride_1,
    src_stride_0,
    src_stride_1,
    output_stride_0,
    output_stride_1,
    output_dim_size,  # Size of output along scatter dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DIM: tl.constexpr,  # Dimension to scatter along (0 or 1)
    REDUCE_MODE: tl.constexpr,  # 0=assign, 1=add, 2=max, 3=min
):
    """
    2D scatter kernel with dimension specification.

    For DIM=0: output[indices[i,j], j] = src[i,j]
    For DIM=1: output[i, indices[i,j]] = src[i,j]
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

    # Load indices and source values
    indices_offsets = offs_m * indices_stride_0 + offs_n * indices_stride_1
    indices = tl.load(indices_ptr + indices_offsets, mask=mask, other=0)

    src_offsets = offs_m * src_stride_0 + offs_n * src_stride_1
    src_values = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    # Bounds check
    valid_indices = (indices >= 0) & (indices < output_dim_size)

    # Compute output offsets based on scatter dimension
    if DIM == 0:
        output_offsets = indices * output_stride_0 + offs_n * output_stride_1
    else:
        output_offsets = offs_m * output_stride_0 + indices * output_stride_1

    scatter_mask = mask & valid_indices

    # Scatter with appropriate reduction mode
    if REDUCE_MODE == 0:
        tl.store(output_ptr + output_offsets, src_values, mask=scatter_mask)
    elif REDUCE_MODE == 1:
        tl.atomic_add(output_ptr + output_offsets, src_values, mask=scatter_mask)
    elif REDUCE_MODE == 2:
        tl.atomic_max(output_ptr + output_offsets, src_values, mask=scatter_mask)
    elif REDUCE_MODE == 3:
        tl.atomic_min(output_ptr + output_offsets, src_values, mask=scatter_mask)


def scatter_triton(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str = None,
) -> torch.Tensor:
    """
    PyTorch wrapper for Triton scatter operation.

    Args:
        input: Input tensor to scatter into
        dim: Dimension along which to scatter
        index: Index tensor
        src: Source values to scatter
        reduce: Reduction mode ('add', 'multiply', None)

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

    # Map reduce mode to integer
    reduce_mode_map = {
        None: 0,
        "add": 1,
        "sum": 1,
        "multiply": 2,
        "amax": 2,  # max
        "amin": 3,  # min
    }
    reduce_mode = reduce_mode_map.get(reduce, 0)

    # For 1D case
    if input.dim() == 1:
        n_elements = index.numel()
        output_size = output.size(0)

        BLOCK_SIZE = 1024
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        if reduce == "add" or reduce == "sum":
            scatter_add_kernel[grid](
                input,
                index,
                src,
                output,
                n_elements,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif reduce == "multiply":
            scatter_mul_kernel[grid](
                input,
                index,
                src,
                output,
                n_elements,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif reduce == "amax":
            scatter_max_kernel[grid](
                input,
                index,
                src,
                output,
                n_elements,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif reduce == "amin":
            scatter_min_kernel[grid](
                input,
                index,
                src,
                output,
                n_elements,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
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

    # For 2D case
    if input.dim() == 2 and index.dim() == 2:
        M, N = index.shape
        output_dim_size = output.size(dim)

        # Ensure contiguous
        input = input.contiguous()
        index = index.contiguous()
        src = src.contiguous()
        output = output.contiguous()

        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32

        # grid = (
        #    triton.cdiv(M, BLOCK_SIZE_M),
        #    triton.cdiv(N, BLOCK_SIZE_N),
        # )
        grid = (1,)

        scatter_2d_kernel[grid](
            input,
            index,
            src,
            output,
            M,
            N,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            src.stride(0),
            src.stride(1),
            output.stride(0),
            output.stride(1),
            output_dim_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DIM=dim,
            REDUCE_MODE=reduce_mode,
        )

        return output

    # For general ND case, fall back to PyTorch
    return input.scatter(dim, index, src, reduce=reduce)


def scatter_triton_(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str = None,
) -> torch.Tensor:
    """
    In-place version of scatter.
    """
    assert input.device == index.device == src.device
    assert index.dtype in [torch.int32, torch.int64]

    if dim < 0:
        dim = input.dim() + dim

    reduce_mode_map = {
        None: 0,
        "add": 1,
        "multiply": 2,
        "amax": 2,
        "amin": 3,
    }
    reduce_mode = reduce_mode_map.get(reduce, 0)

    # 1D case
    if input.dim() == 1:
        n_elements = index.numel()
        output_size = input.size(0)

        BLOCK_SIZE = 1024
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        grid = (1,)

        if reduce == "add":
            scatter_add_kernel[grid](
                input, index, src, input, n_elements, output_size, BLOCK_SIZE=BLOCK_SIZE
            )
        elif reduce == "amax":
            scatter_max_kernel[grid](
                input, index, src, input, n_elements, output_size, BLOCK_SIZE=BLOCK_SIZE
            )
        elif reduce == "amin":
            scatter_min_kernel[grid](
                input, index, src, input, n_elements, output_size, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            scatter_kernel[grid](
                input, index, src, input, n_elements, output_size, BLOCK_SIZE=BLOCK_SIZE
            )

        return input

    # 2D case
    if input.dim() == 2:
        M, N = index.shape
        output_dim_size = input.size(dim)

        input_contig = input.contiguous()
        index_contig = index.contiguous()
        src_contig = src.contiguous()

        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32

        # grid = (
        #    triton.cdiv(M, BLOCK_SIZE_M),
        #    triton.cdiv(N, BLOCK_SIZE_N),
        # )
        grid = (1,)

        scatter_2d_kernel[grid](
            input_contig,
            index_contig,
            src_contig,
            input_contig,
            M,
            N,
            input_contig.stride(0),
            input_contig.stride(1),
            index_contig.stride(0),
            index_contig.stride(1),
            src_contig.stride(0),
            src_contig.stride(1),
            input_contig.stride(0),
            input_contig.stride(1),
            output_dim_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DIM=dim,
            REDUCE_MODE=reduce_mode,
        )

        if not input.is_contiguous():
            input.copy_(input_contig)

        return input

    # Fall back to PyTorch
    return input.scatter_(dim, index, src, reduce=reduce)


@triton.jit
def scatter_reduce_sorted_kernel(
    indices_ptr,
    src_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized scatter-add for sorted indices.
    Can avoid atomics by processing sequentially within blocks.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices and values
    indices = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    # For sorted indices, we can aggregate within the block
    # This is more complex - simplified version uses atomics
    valid = mask & (indices >= 0) & (indices < output_size)
    tl.atomic_add(output_ptr + indices, values, mask=valid)


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_scatter_1d():
    input = torch.zeros(5)
    indices = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int64)
    source = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    output = scatter_triton(input, 0, indices, source)
    reference = input.scatter(0, indices, source)

    assert torch.allclose(output, reference)


@dsp
@parametrize_dims(0, 1)
def test_scatter_2d(dim):
    input = torch.zeros(5, 4)
    indices = torch.randint(0, 3, (3, 4), dtype=torch.int64)
    source = torch.randn(3, 4)

    output = scatter_triton(input, dim, indices, source)
    reference = input.scatter(dim, indices, source)

    assert torch.allclose(output, reference)


@fixme
@dsp
@parametrize_reduces("sum", "amax")
def test_scatter_reduce(reduce):
    input = torch.ones(5)
    indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    source = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    output = scatter_triton(input, 0, indices, source, reduce=reduce)
    reference = input.scatter_reduce(0, indices, source, reduce=reduce)

    assert torch.allclose(output, reference)


@dsp
def test_scatter_inplace():
    reference = torch.zeros(5)
    output = torch.zeros(5)
    indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    source = torch.tensor([1.0, 2.0, 3.0])

    reference.scatter_(0, indices, source)
    scatter_triton_(output, 0, indices, source)

    assert torch.allclose(output, reference)


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
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark scatter-add (common in gradient accumulation)
    print("\n1D Scatter-Add Benchmark:")
    for size in sizes:
        input = torch.zeros(size)
        # Create duplicate indices for realistic scatter-add scenario
        indices = torch.randint(0, size // 10, (size,), dtype=torch.int64)
        src = torch.randn(size)

        # Warmup
        for _ in range(10):
            _ = input.scatter(0, indices, src, reduce="add")
            _ = scatter_triton(input, 0, indices, src, reduce="add")


        n_iters = 100

        start = time.time()
        for _ in range(n_iters):
            out = input.scatter(0, indices, src, reduce="add")
        time_torch = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            out = scatter_triton(input, 0, indices, src, reduce="add")
        time_triton = (time.time() - start) / n_iters * 1000

        print(
            f"Size {size:>7}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )

    # Benchmark 2D scatter
    print("\n2D Scatter Benchmark:")
    shapes = [(100, 100), (512, 512), (1024, 1024)]

    for M, N in shapes:
        input = torch.zeros(M, N)
        indices = torch.randint(0, M, (M, N), dtype=torch.int64)
        src = torch.randn(M, N)

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
            f"Shape {M}x{N}: PyTorch {time_torch:.4f} ms | Triton {time_triton:.4f} ms | Speedup {time_torch / time_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark_scatter()
