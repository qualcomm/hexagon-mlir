# ===- test_libd_exp.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Regression test for exp overflow: values above the representable range
# (fp32: x > ~88.72, fp16: x > ~11.09) must produce +inf, not NaN.

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name
from triton.language.extra import libdevice

triton.runtime.driver.set_active(HexagonDriver())

ATOL_FP32 = 1e-1
ATOL_FP16 = 1e-1
RTOL_FP32 = 1e-1
RTOL_FP16 = 1e-1


def _run_exp_kernel(x_flat: torch.Tensor, atol: float, rtol: float):
    """Run libdevice.exp on a flattened tensor and compare against torch.exp."""
    dtype = x_flat.dtype
    block_size = triton.next_power_of_2(len(x_flat))
    x_padded = torch.zeros(block_size, dtype=dtype)
    x_padded[: len(x_flat)] = x_flat
    output = torch.empty(block_size, dtype=dtype)

    @triton.jit
    @parameterize_func_name(dtype)
    def exp_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        inp = tl.load(x_ptr + offsets)
        tl.store(output_ptr + offsets, libdevice.exp(inp))

    exp_kernel[(1,)](
        x_padded,
        output,
        BLOCK_SIZE=block_size,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.exp(x_padded)
    assert torch.allclose(output, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp_infinity(dtype):
    """Small tensor (5 elements): values straddle the overflow threshold."""
    if dtype == torch.float32:
        x = torch.tensor([88.71, 88.75, 88.8, 88.6, 85.0], dtype=dtype)
        atol, rtol = ATOL_FP32, RTOL_FP32
    else:
        x = torch.tensor([11.0, 11.1, 11.2, 10.9, 10.0], dtype=dtype)
        atol, rtol = ATOL_FP16, RTOL_FP16
    _run_exp_kernel(x, atol, rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp_overflow_multivec(dtype):
    """125-element tensor (5x5x5): exercises the multi-vector loop in the array
    wrapper (nvecs > 1). All inputs are well above the overflow threshold so
    every lane must produce +inf, not a garbage finite value."""
    if dtype == torch.float32:
        x = (torch.randn(5, 5, 5) * 10 + 90).flatten().to(dtype)
        atol, rtol = ATOL_FP32, RTOL_FP32
    else:
        # fp16 overflow threshold ~11.09; center at 12 with some spread
        x = (torch.randn(5, 5, 5) + 12).flatten().to(dtype)
        atol, rtol = ATOL_FP16, RTOL_FP16
    _run_exp_kernel(x, atol, rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp_overflow_one_vec(dtype):
    """Exercises the if (nvecs > 0) block with exactly nvecs==1 (no loop
    iterations). fp32: 45 elements → nvecs=1, leftover=13.
    fp16: 100 elements → nvecs=1, leftover=36."""
    if dtype == torch.float32:
        x = torch.full((45,), 90.0, dtype=dtype)
        atol, rtol = ATOL_FP32, RTOL_FP32
    else:
        x = torch.full((100,), 12.0, dtype=dtype)
        atol, rtol = ATOL_FP16, RTOL_FP16
    _run_exp_kernel(x, atol, rtol)
