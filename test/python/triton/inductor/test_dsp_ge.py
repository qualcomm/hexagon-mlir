# ===- test_dsp_ge.py -------------------------------------------------------===
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

from .. import dsp, parametrize_dtypes, parametrize_shapes

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def ge_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform greater than or equal comparison
    result = x >= y

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def ge_scalar_kernel(
    x_ptr,
    output_ptr,
    scalar_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Perform comparison with scalar
    result = x >= scalar_value

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def ge_2d_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    x_stride_m,
    x_stride_n,
    y_stride_m,
    y_stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block offsets
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)

    # Create 2D offset grid
    m_offsets_2d = m_offsets[:, None]
    n_offsets_2d = n_offsets[None, :]

    # Create mask
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    # Calculate memory offsets
    x_offsets = m_offsets_2d * x_stride_m + n_offsets_2d * x_stride_n
    y_offsets = m_offsets_2d * y_stride_m + n_offsets_2d * y_stride_n
    output_offsets = m_offsets_2d * output_stride_m + n_offsets_2d * output_stride_n

    # Load values
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)

    # Perform comparison
    result = x >= y

    # Store result
    tl.store(output_ptr + output_offsets, result, mask=mask)


@triton.jit
def ge_broadcast_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    y_is_scalar: tl.constexpr,
    y_scalar_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load or use scalar y
    if y_is_scalar:
        y = y_scalar_value
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform comparison
    result = x >= y

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def ge_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Handle scalar y
    if isinstance(y, (int, float)):
        return ge_triton_scalar(x, y)

    # Ensure same shape
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    # Flatten for processing
    x_flat = x.flatten().contiguous()
    y_flat = y.flatten().contiguous()
    n_elements = x_flat.numel()

    # Create output tensor
    output = torch.empty(n_elements, dtype=torch.bool, device=x.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    ge_kernel[grid](
        x_flat,
        y_flat,
        output,
        n_elements,
        BLOCK_SIZE,
    )

    # Reshape to original shape
    return output.reshape(x.shape)


def ge_triton_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor:
    x_flat = x.flatten().contiguous()
    n_elements = x_flat.numel()

    # Create output tensor
    output = torch.empty(n_elements, dtype=torch.bool, device=x.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    ge_scalar_kernel[grid](
        x_flat,
        output,
        scalar,
        n_elements,
        BLOCK_SIZE,
    )

    return output.reshape(x.shape)


def ge_triton_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.ndim >= 2 and y.ndim >= 2, "Inputs must be at least 2D"

    # Reshape to 2D if needed
    orig_shape = x.shape
    if x.ndim > 2:
        x_2d = x.reshape(-1, x.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])
    else:
        x_2d = x
        y_2d = y

    x_2d = x_2d.contiguous()
    y_2d = y_2d.contiguous()

    M, N = x_2d.shape

    # Create output
    output = torch.empty_like(x_2d, dtype=torch.bool)

    # Launch kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    ge_2d_kernel[grid](
        x_2d,
        y_2d,
        output,
        M,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    return output.reshape(orig_shape)


# Additional comparison operators for completeness
@triton.jit
def gt_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Less than: x < y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x < y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def ge_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Greater than or equal: x >= y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x >= y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gt_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Greater than: x > y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x > y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def eq_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Equal: x == y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x == y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def ne_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Not equal: x != y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x != y
    tl.store(output_ptr + offsets, result, mask=mask)


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
@parametrize_shapes((1024,), (10 * 1024 * 1024,))
def test_ge_1d(shape):
    x = torch.randn(shape)
    y = torch.randn(shape)

    output = ge_triton(x, y)
    reference = torch.ge(x, y)

    assert torch.equal(output, reference)


@dsp
def test_ge_scalar():
    x = torch.randn(1024)
    scalar = 0.5

    output = ge_triton_scalar(x, scalar)
    reference = torch.ge(x, scalar)

    assert torch.equal(output, reference)


@dsp
def test_ge_2d():
    x = torch.randn(256, 512)
    y = torch.randn(256, 512)

    output = ge_triton_2d(x, y)
    reference = torch.ge(x, y)

    assert torch.equal(output, reference)


@dsp
def test_ge_all_equal():
    x = torch.ones(128)
    y = torch.ones(128)

    output = ge_triton(x, y)
    reference = torch.ones(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_ge_all_less():
    x = torch.zeros(128)
    y = torch.ones(128)

    output = ge_triton(x, y)
    reference = torch.zeros(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_ge_all_greater():
    x = torch.ones(128)
    y = torch.zeros(128)

    output = ge_triton(x, y)
    reference = torch.ones(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_ge_inf_nan():
    x = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 0.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])

    output = ge_triton(x, y)
    reference = torch.ge(x, y)

    assert torch.equal(output, reference)


@dsp
@parametrize_dtypes(torch.float16, torch.float32)
def test_ge_dtypes(dtype):
    x = torch.randn(128, dtype=dtype)
    y = torch.randn(128, dtype=dtype)

    output = ge_triton(x, y)
    reference = torch.ge(x, y)

    assert torch.equal(output, reference)


@dsp
@parametrize_shapes((10, 20), (5, 10, 15), (2, 3, 4, 5))
def test_ge_shapes(shape):
    x = torch.randn(shape)
    y = torch.randn(shape)

    output = ge_triton(x, y)
    reference = torch.ge(x, y)

    assert torch.equal(output, reference)
