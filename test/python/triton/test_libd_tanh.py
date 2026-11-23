# ===- test_libd_tanh.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name
from triton.language.extra import libdevice

triton.runtime.driver.set_active(HexagonDriver())

NUM_THREADS = 4
NUM_ELEMENTS = 8192
BLOCK_SIZE = NUM_ELEMENTS // NUM_THREADS

# libdevice.tanh maps to the stock QHL tanh extern (qhmath_hvx_tanh_af/_ahf).
# fp32 holds ~1e-5; fp16 adds rounding so it uses a looser 1e-2.
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_tanh(dtype):

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_tanh_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.tanh(x)
        tl.store(output_ptr + offsets, output)

    # Sample [-4, 4]: tanh saturates to +/-1 outside this, so this covers the
    # transition band where accuracy matters most.
    x = torch.linspace(-4.0, 4.0, NUM_ELEMENTS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    libd_tanh_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.tanh(x)
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_tanh_extremes(dtype):
    """Saturation and infinity behaviour.

    Two things must hold:
      * Large |x| saturates to +/-1.
      * The infinities saturate: tanh(+inf)=1, tanh(-inf)=-1.

    NaN is deliberately NOT tested: libdevice.tanh maps to the stock QHL tanh
    extern, which returns +/-1 for a NaN input rather than propagating it. NaN
    propagation is handled one level up in libdevice.erf (guarded on the
    original input), not in the shared tanh extern.
    """

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_tanh_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.tanh(x)
        tl.store(output_ptr + offsets, output)

    # Saturation-scale magnitudes, the transition band, and the infinities.
    # Padded with in-range zeros so the length stays a power of two for
    # tl.arange in the kernel.
    vals = [
        -1000.0,
        -100.0,
        -8.0,
        -1.0,
        -1e-3,
        0.0,
        1e-3,
        1.0,
        8.0,
        100.0,
        1000.0,
        float("inf"),
        float("-inf"),
    ]
    vals += [0.0] * (32 - len(vals))
    assert len(vals) == 32

    x = torch.tensor(vals, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    libd_tanh_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=x.numel(),
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.tanh(x)

    assert torch.isfinite(output.float()).all(), f"non-finite tanh output: {output}"
    assert torch.allclose(
        output,
        reference,
        atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_tanh_nan(dtype):
    """NaN behavior: tanh does NOT propagate NaN.

    The stock QHL tanh extern (qhmath_hvx_tanh_af/_ahf) returns +/-1 for a NaN
    input rather than propagating it. This is expected behavior for the shared
    tanh extern. NaN propagation is handled one level up in libdevice.erf
    (guarded on the original input), not in the shared tanh extern.
    """

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_tanh_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.tanh(x)
        tl.store(output_ptr + offsets, output)

    # Test NaN input: tanh(nan) returns +/-1 (not nan).
    x = torch.tensor([float("nan")], dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    libd_tanh_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=x.numel(),
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    # tanh(nan) should saturate to +/-1 (magnitude in [0.99, 1.0001]), not nan.
    abs_out = torch.abs(output.float())
    assert torch.isfinite(output).all(), f"tanh(nan) should be finite: {output}"
    assert (
        (abs_out >= 0.99) & (abs_out <= 1.0001)
    ).all(), f"tanh(nan) should saturate to +/-1: {output}"
