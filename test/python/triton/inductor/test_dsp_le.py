# ===- test_dsp_le.py -------------------------------------------------------===
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
def le_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise less than or equal to comparison: x <= y

    Returns True where x[i] <= y[i], False otherwise.

    Args:
        x_ptr: First input tensor
        y_ptr: Second input tensor (or scalar broadcast)
        output_ptr: Output boolean tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements per block
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform less than or equal comparison
    result = x <= y

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def le_scalar_kernel(
    x_ptr,
    output_ptr,
    scalar_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise less than or equal to comparison with scalar: x <= scalar

    Args:
        x_ptr: Input tensor
        scalar_value: Scalar value to compare against
        output_ptr: Output boolean tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements per block
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Perform comparison with scalar
    result = x <= scalar_value

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def le_2d_kernel(
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
    """
    2D optimized less than or equal to comparison.
    Better memory coalescing for 2D tensors.

    Args:
        x_ptr: First input tensor
        y_ptr: Second input tensor
        output_ptr: Output boolean tensor
        M: Number of rows
        N: Number of columns
        *_stride_m/n: Strides for each dimension
        BLOCK_SIZE_M/N: Block sizes
    """
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
    result = x <= y

    # Store result
    tl.store(output_ptr + output_offsets, result, mask=mask)


@triton.jit
def le_broadcast_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    y_is_scalar: tl.constexpr,
    y_scalar_value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Less than or equal comparison with broadcasting support.

    Args:
        x_ptr: First input tensor
        y_ptr: Second input tensor (ignored if y_is_scalar=True)
        output_ptr: Output boolean tensor
        n_elements: Total number of elements
        y_is_scalar: Whether y is a scalar value
        y_scalar_value: Scalar value for y (used if y_is_scalar=True)
        BLOCK_SIZE: Number of elements per block
    """
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
    result = x <= y

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def le_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of torch.le (less than or equal).

    Returns a boolean tensor where each element indicates if x[i] <= y[i].

    Args:
        x: First input tensor
        y: Second input tensor or scalar

    Returns:
        Boolean tensor of same shape as x
    """
    # Handle scalar y
    if isinstance(y, (int, float)):
        return le_triton_scalar(x, y)

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

    le_kernel[grid](
        x_flat,
        y_flat,
        output,
        n_elements,
        BLOCK_SIZE,
    )

    # Reshape to original shape
    return output.reshape(x.shape)


def le_triton_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """
    Triton implementation of x <= scalar.

    Args:
        x: Input tensor
        scalar: Scalar value to compare against

    Returns:
        Boolean tensor of same shape as x
    """
    x_flat = x.flatten().contiguous()
    n_elements = x_flat.numel()

    # Create output tensor
    output = torch.empty(n_elements, dtype=torch.bool, device=x.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    le_scalar_kernel[grid](
        x_flat,
        output,
        scalar,
        n_elements,
        BLOCK_SIZE,
    )

    return output.reshape(x.shape)


def le_triton_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Optimized 2D version with better memory coalescing.

    Args:
        x: First input tensor (2D)
        y: Second input tensor (2D)

    Returns:
        Boolean tensor indicating x <= y
    """
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

    le_2d_kernel[grid](
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
def lt_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
def test_le_1d(shape):
    x = torch.randn(shape)
    y = torch.randn(shape)

    output = le_triton(x, y)
    reference = torch.le(x, y)

    assert torch.equal(output, reference)


@dsp
def test_le_scalar():
    x = torch.randn(1024)
    scalar = 0.5

    output = le_triton_scalar(x, scalar)
    reference = torch.le(x, scalar)

    assert torch.equal(output, reference)


@dsp
def test_le_2d():
    x = torch.randn(256, 512)
    y = torch.randn(256, 512)

    output = le_triton_2d(x, y)
    reference = torch.le(x, y)

    assert torch.equal(output, reference)


@dsp
def test_le_all_equal():
    x = torch.ones(128)
    y = torch.ones(128)

    output = le_triton(x, y)
    reference = torch.ones(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_le_all_less():
    x = torch.zeros(128)
    y = torch.ones(128)

    output = le_triton(x, y)
    reference = torch.ones(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_le_all_greater():
    x = torch.ones(128)
    y = torch.zeros(128)

    output = le_triton(x, y)
    reference = torch.zeros(128, dtype=torch.bool)

    assert torch.equal(output, reference)


@dsp
def test_le_inf_nan():
    x = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 0.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])

    output = le_triton(x, y)
    reference = torch.le(x, y)

    assert torch.equal(output, reference)


@dsp
@parametrize_dtypes(torch.float16, torch.float32)
def test_le_dtypes(dtype):
    x = torch.randn(128, dtype=dtype)
    y = torch.randn(128, dtype=dtype)

    output = le_triton(x, y)
    reference = torch.le(x, y)

    assert torch.equal(output, reference)


@dsp
@parametrize_shapes((10, 20), (5, 10, 15), (2, 3, 4, 5))
def test_le_shapes(shape):
    x = torch.randn(shape)
    y = torch.randn(shape)

    output = le_triton(x, y)
    reference = torch.le(x, y)

    assert torch.equal(output, reference)
