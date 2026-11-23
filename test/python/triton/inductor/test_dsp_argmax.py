# ===- test_dsp_argmax.py ---------------------------------------------------===
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
def argmax_1d_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D argmax kernel - finds the index of maximum value in a 1D array.
    """
    # Each program handles one block
    pid = tl.program_id(0)

    # Initialize with first element
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

    # Find max value and its index within block
    max_val = tl.max(values, axis=0)
    max_idx = tl.argmax(values, axis=0)

    # Store block-level results
    # Note: This is a simplified version; for full reduction across blocks,
    # we'd need a second kernel or atomic operations
    if pid == 0:
        tl.store(output_ptr, block_start + max_idx)


@triton.jit
def argmax_reduction_kernel(
    input_ptr,
    output_ptr,
    M,  # Number of rows
    N,  # Number of columns (reduction dimension)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Argmax along the last dimension using reduction.
    Each row is processed independently.
    """
    # Program ID for row
    pid_m = tl.program_id(0)

    # Starting position for this row
    row_start = pid_m * N

    # Initialize max tracking
    running_max = -float("inf")
    running_idx = 0

    # Iterate over the reduction dimension in blocks
    for block_start_n in range(0, N, BLOCK_SIZE_N):
        offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < N

        # Load values
        vals = tl.load(
            input_ptr + row_start + offsets_n, mask=mask_n, other=-float("inf")
        )

        # Find max in this block
        block_max = tl.max(vals, axis=0)

        # Update global max if needed
        if block_max > running_max:
            # Find which element in block is the max
            block_argmax = tl.argmax(vals, axis=0)
            running_max = block_max
            running_idx = block_start_n + block_argmax

    # Store result
    tl.store(output_ptr + pid_m, running_idx)


@triton.jit
def argmax_2d_kernel(
    input_ptr,
    output_ptr,
    stride_m,
    stride_n,
    M,  # Size of dimension 0
    N,  # Size of dimension 1 (reduction dimension)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized 2D argmax kernel with coalesced memory access.
    Reduces along dimension 1 (columns).
    """
    pid_m = tl.program_id(0)

    # Calculate row indices for this program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M

    # Initialize max values and indices for all rows in this block
    running_max = tl.full([BLOCK_SIZE_M], -float("inf"), dtype=tl.float32)
    running_idx = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)

    # Iterate over columns
    for n_block in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_block + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # Create 2D offset pattern
        m_expanded = m_offsets[:, None]
        n_expanded = n_offsets[None, :]

        # Calculate memory offsets
        offsets = m_expanded * stride_m + n_expanded * stride_n
        mask = m_mask[:, None] & n_mask[None, :]

        # Load values
        vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

        # Find max along N dimension for each M
        block_max = tl.max(vals, axis=1)

        # Update running max
        for i in tl.static_range(BLOCK_SIZE_M):
            if block_max[i] > running_max[i]:
                # Find the index within this block
                row_vals = vals[i, :]
                local_argmax = tl.argmax(row_vals, axis=0)
                running_max[i] = block_max[i]
                running_idx[i] = n_block + local_argmax

    # Store results
    output_offsets = m_offsets
    tl.store(output_ptr + output_offsets, running_idx, mask=m_mask)


@triton.jit
def argmax_keepdim_kernel(
    input_ptr,
    output_ptr,
    stride_batch,
    stride_m,
    stride_n,
    batch_size,
    M,
    N,  # Reduction dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Argmax with batch support for higher dimensional tensors.
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Calculate batch offset
    batch_offset = pid_batch * stride_batch

    # Row offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M

    # Initialize
    running_max = tl.full([BLOCK_SIZE_M], -float("inf"), dtype=tl.float32)
    running_idx = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)

    # Iterate over reduction dimension
    for n_block in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_block + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # 2D access pattern
        m_expanded = m_offsets[:, None]
        n_expanded = n_offsets[None, :]

        offsets = batch_offset + m_expanded * stride_m + n_expanded * stride_n
        mask = m_mask[:, None] & n_mask[None, :]

        vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

        # Reduce along columns
        block_max = tl.max(vals, axis=1)

        for i in tl.static_range(BLOCK_SIZE_M):
            if block_max[i] > running_max[i]:
                row_vals = vals[i, :]
                local_argmax = tl.argmax(row_vals, axis=0)
                running_max[i] = block_max[i]
                running_idx[i] = n_block + local_argmax

    # Store output
    output_offsets = pid_batch * M + m_offsets
    tl.store(output_ptr + output_offsets, running_idx, mask=m_mask)


def argmax_triton(x: torch.Tensor, dim: int = None, keepdim: bool = False):
    """
    Triton implementation of torch.argmax.

    Args:
        x: Input tensor
        dim: Dimension along which to find argmax. If None, returns argmax of flattened tensor.
        keepdim: Whether to keep the reduced dimension

    Returns:
        Tensor containing indices of maximum values
    """
    if dim is None:
        # Flatten and find global argmax
        x_flat = x.flatten()
        n_elements = x_flat.numel()

        # For simplicity, use PyTorch for global argmax
        # A full Triton implementation would need multi-stage reduction
        return torch.argmax(x_flat)

    # Normalize dimension
    if dim < 0:
        dim = x.ndim + dim

    # Move reduction dimension to last position for easier processing
    x_transposed = x.movedim(dim, -1).contiguous()
    orig_shape = x_transposed.shape

    # Reshape to 2D: [batch_size, reduction_size]
    batch_size = 1
    for i in range(len(orig_shape) - 1):
        batch_size *= orig_shape[i]

    reduction_size = orig_shape[-1]
    x_2d = x_transposed.reshape(batch_size, reduction_size)

    # Prepare output
    output = torch.empty(batch_size, dtype=torch.int32, device=x.device)

    # Launch kernel
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1024

    grid = (batch_size,)
    # grid = (1,)

    argmax_reduction_kernel[grid](
        x_2d,
        output,
        batch_size,
        reduction_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    # Reshape output
    if keepdim:
        output_shape = list(orig_shape[:-1]) + [1]
        output = output.reshape(output_shape)
        # Move dimension back
        output = output.movedim(-1, dim)
    else:
        output_shape = list(orig_shape[:-1])
        if len(output_shape) == 0:
            output = output.squeeze()
        else:
            output = output.reshape(output_shape)
            # Move dimensions back (accounting for removed dimension)
            if dim < x.ndim - 1:
                # Need to adjust for the removed dimension
                pass

    return output.long()


def argmax_triton_optimized(x: torch.Tensor, dim: int = -1):
    """
    Optimized version for common case of reducing along last dimension.
    """
    if dim != -1 and dim != x.ndim - 1:
        x = x.movedim(dim, -1)

    # Get shape
    *batch_dims, reduction_size = x.shape
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    # Reshape to 2D
    x_2d = x.reshape(batch_size, reduction_size).contiguous()
    output = torch.empty(batch_size, dtype=torch.int32, device=x.device)

    # Kernel configuration
    BLOCK_SIZE_N = min(1024, triton.next_power_of_2(reduction_size))
    grid = (batch_size,)
    # grid = (1,)

    argmax_reduction_kernel[grid](
        x_2d,
        output,
        batch_size,
        reduction_size,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    # Reshape back
    if len(batch_dims) > 0:
        output = output.reshape(batch_dims)
    else:
        output = output.squeeze()

    return output.long()


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_argmax_2d():
    x = torch.randn(128, 256)
    output = argmax_triton_optimized(x, dim=-1)
    reference = torch.argmax(x, dim=-1)

    assert torch.equal(output, reference)


@dsp
def test_argmax_3d():
    x = torch.randn(32, 64, 128)
    output = argmax_triton_optimized(x, dim=-1)
    reference = torch.argmax(x, dim=-1)

    assert torch.equal(output, reference)


@dsp
def test_argmax_edge_cases():
    x = torch.tensor([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]])
    output = argmax_triton_optimized(x, dim=-1)
    reference = torch.argmax(x, dim=-1)

    assert torch.equal(output, reference)


@dsp
@parametrize_dims(0, 1, 2, -1)
def test_argmax(dim):
    x = torch.randn(16, 32, 64)
    output = argmax_triton(x, dim=dim)
    reference = torch.argmax(x, dim=dim)

    assert torch.equal(output, reference)


def benchmark():
    import time

    x = torch.randn(1024, 4096)

    print(f"\nTest 4 - Benchmarking (1024, 4096):")

    # Warmup
    for _ in range(1):
        _ = argmax_triton_optimized(x, dim=-1)
        _ = torch.argmax(x, dim=-1)

    # Benchmark Triton
    start = time.time()
    for _ in range(1):
        result_triton = argmax_triton_optimized(x, dim=-1)
    triton_time = (time.time() - start) / 100

    # Benchmark PyTorch
    start = time.time()
    for _ in range(100):
        result_torch = torch.argmax(x, dim=-1)
    torch_time = (time.time() - start) / 100

    print(f"Triton time: {triton_time * 1000:.3f} ms")
    print(f"PyTorch time: {torch_time * 1000:.3f} ms")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"Results match: {torch.equal(result_triton, result_torch)}")


if __name__ == "__main__":
    benchmark()
