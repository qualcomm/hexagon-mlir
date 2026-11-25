# ===- test_libd_erfinv.py --------------------------------------------------===
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

# libdevice.erfinv is a single-path implementation (Pavlis rational
# approximation + one guarded Newton step). Measured max abs
# error vs true erfinv is ~1.7e-3 over (-0.99, 0.99) (peaks near |y|~0.8 where
# the Newton step rides the erf-polynomial branch switch), so fp32 uses 2e-3;
# fp16 rounding pushes that to 1e-2.
ATOL_FP32 = 2e-3
ATOL_FP16 = 1e-2


def _atol(dtype):
    return ATOL_FP16 if dtype == torch.float16 else ATOL_FP32


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_erfinv(dtype):

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_erfinv_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        # erfinv has no native HVX symbol; libdevice.erfinv implements the
        # Pavlis rational approximation. Exercise it directly so the
        # shared-libdevice change is covered where it is implemented (also
        # relevant for DSP).
        output = libdevice.erfinv(x)
        tl.store(output_ptr + offsets, output)

    # erfinv is finite only on (-1, 1); sample just inside the boundary so the
    # steep-slope region near |y|=1 is covered. The 8192-point linspace spacing
    # (~2.4e-4) also densely covers the |y|~0.8 band where the Newton refinement
    # step is most active, so the tolerance check below exercises it directly.
    x = torch.linspace(-0.99, 0.99, NUM_ELEMENTS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    libd_erfinv_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.erfinv(x)
    assert torch.allclose(output, reference, atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_erfinv_extremes(dtype):
    """Boundary, out-of-domain, near-zero, and NaN behaviour.

    erfinv is defined on (-1, 1). Beyond the finite ramp, four things must hold:
      * y = +/-1 saturates: erfinv(1)=+inf, erfinv(-1)=-inf.
      * |y| > 1 is out of domain: erfinv(y)=nan.
      * Small y stays near erfinv(y) ~= sqrt(pi)/2 * y.
      * NaN propagates: erfinv(nan)=nan.
    """

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_erfinv_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.erfinv(x)
        tl.store(output_ptr + offsets, output)

    # Finite domain probes, the +/-1 boundary, out-of-domain magnitudes, and a
    # NaN. Padded with in-range zeros so the length stays a power of two for
    # tl.arange in the kernel.
    vals = [
        -0.99,
        -0.5,
        -1e-3,
        0.0,
        1e-3,
        0.5,
        0.9,
        0.99,
        1.0,
        -1.0,
        1.5,
        -2.0,
        float("nan"),
    ]
    vals += [0.0] * (32 - len(vals))
    assert len(vals) == 32

    x = torch.tensor(vals, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    libd_erfinv_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=x.numel(),
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.erfinv(x)

    # NaN must propagate; allclose doesn't handle NaN, so mask it out first.
    nan_mask = torch.isnan(reference)
    assert torch.isnan(
        output.float()[nan_mask]
    ).all(), f"erfinv did not propagate NaN: {output}"

    finite_mask = torch.isfinite(reference)
    assert torch.allclose(
        output[finite_mask],
        reference[finite_mask],
        atol=_atol(dtype),
    )

    # y = +/-1 -> +/-inf (with the right sign).
    inf_mask = torch.isinf(reference)
    assert torch.equal(
        output.float()[inf_mask], reference.float()[inf_mask]
    ), f"erfinv boundary did not saturate to +/-inf: {output}"
