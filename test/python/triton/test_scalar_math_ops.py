# ===- test_scalar_math_ops.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Regression test for commit 5906d30d: LowerScalarMathPattern in
# LowerLibdevicePass.
#
# For 1-element tensors, TritonReorderBroadcast sinks tt.splat past math.*
# ops, producing a bare scalar math.sin/cos/etc with no enclosing
# linalg.generic.  Without the fix those scalars lower to llvm.intr.sin →
# sinf@plt which is unmapped in the DSP JIT and triggers a ~14 s DSP
# watchdog timeout (Error code: 500).
#
# This test exercises the fixed path end-to-end on a 1-element (size-1)
# tensor for the ops that were registered in the patch.

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

ATOL = 1e-2


def _run_unary(kernel, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    kernel[(1,)](
        x,
        out,
        BLOCK_SIZE=1,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    return out


@triton.jit
def sin_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.sin(x))


@triton.jit
def cos_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.cos(x))


@triton.jit
def exp_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.exp(x))


@triton.jit
def silu_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x * tl.sigmoid(x))


@triton.jit
def sqrt_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.sqrt(x))


@triton.jit
def rsqrt_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.math.rsqrt(x))


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_sin_size1(dtype):
    x = torch.tensor([0.5], dtype=dtype)
    out = _run_unary(sin_kernel, x)
    ref = torch.sin(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"sin size-1 failed: out={out}, ref={ref}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_cos_size1(dtype):
    x = torch.tensor([0.5], dtype=dtype)
    out = _run_unary(cos_kernel, x)
    ref = torch.cos(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"cos size-1 failed: out={out}, ref={ref}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_exp_size1(dtype):
    x = torch.tensor([1.0], dtype=dtype)
    out = _run_unary(exp_kernel, x)
    ref = torch.exp(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"exp size-1 failed: out={out}, ref={ref}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_silu_size1(dtype):
    """silu(x) = x * sigmoid(x); the sigmoid path exercises the exp scalar path."""
    x = torch.tensor([1.0], dtype=dtype)
    out = _run_unary(silu_kernel, x)
    ref = torch.nn.functional.silu(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"silu size-1 failed: out={out}, ref={ref}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_sqrt_size1(dtype):
    x = torch.tensor([4.0], dtype=dtype)
    out = _run_unary(sqrt_kernel, x)
    ref = torch.sqrt(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"sqrt size-1 failed: out={out}, ref={ref}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scalar_rsqrt_size1(dtype):
    x = torch.tensor([4.0], dtype=dtype)
    out = _run_unary(rsqrt_kernel, x)
    ref = torch.rsqrt(x)
    assert torch.allclose(
        out, ref, atol=ATOL
    ), f"rsqrt size-1 failed: out={out}, ref={ref}"
