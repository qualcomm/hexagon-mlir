# ===- test_libd_erf.py -----------------------------------------------------===
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

# libdevice.erf has two implementations, selected per-call by the
# use_tanh_approx constexpr (default resolved from HEXAGON_ENABLE_ACCURACY at
# import time). The tests pin the constexpr explicitly so both paths are covered
# in one process regardless of the env.
#
# tanh path (use_tanh_approx=True): erf(x) ~= tanh(a*x + b*x**3), max abs error
# vs true erf ~3.6e-4 over [-4, 4]. 1e-5 is unachievable here by ~360x; 2e-3
# keeps headroom above the ~3.6e-4 floor while still catching a broken approx.
# fp16 adds rounding on top, so it uses a looser 1e-2.
ATOL_TANH_FP32 = 2e-3
ATOL_TANH_FP16 = 1e-2

# accurate path (use_tanh_approx=False): a direct port of the eager erf_sf
# polynomial. Measured max abs error vs true erf is ~8e-4 (the polynomial peaks
# near |x|=2 and at the 3.925 saturation clamp, plus on-device qf32 rounding),
# so 1e-3 keeps headroom over that floor while still being tighter and more
# eager-consistent than the tanh path. fp16 rounding near the +/-1 tails is
# ~9.8e-4 (one fp16 ULP), so fp16 keeps the same 1e-2 as the tanh path.
ATOL_ACCURATE_FP32 = 1e-3
ATOL_ACCURATE_FP16 = 1e-2


def _atol(use_tanh_approx, dtype):
    if use_tanh_approx:
        return ATOL_TANH_FP16 if dtype == torch.float16 else ATOL_TANH_FP32
    return ATOL_ACCURATE_FP16 if dtype == torch.float16 else ATOL_ACCURATE_FP32


@pytest.mark.parametrize("use_tanh_approx", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_erf(dtype, use_tanh_approx):

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_erf_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
        USE_TANH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        # erf has no native HVX symbol; libdevice.erf is either a tanh composite
        # (perf) or the accurate eager erf_sf polynomial. Exercise it
        # directly here so the shared-libdevice change is covered where it is
        # implemented.
        output = libdevice.erf(x, use_tanh_approx=USE_TANH)
        tl.store(output_ptr + offsets, output)

    # Sample [-4, 4] so the erf transition band (where accuracy matters most)
    # is well covered rather than only the saturated tails.
    x = torch.linspace(-4.0, 4.0, NUM_ELEMENTS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    libd_erf_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_TANH=use_tanh_approx,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.erf(x)
    assert torch.allclose(output, reference, atol=_atol(use_tanh_approx, dtype))


@pytest.mark.parametrize("use_tanh_approx", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_erf_extremes(dtype, use_tanh_approx):
    """Saturation, near-zero, infinity, and NaN behaviour.

    Beyond the [-4, 4] transition band, four things must hold on both paths:
      * Large |x| saturates to +/-1. libdevice.erf promotes to fp32 first, so
        even fp16 inputs like +/-100 don't overflow the polynomial; the
        composite/polynomial saturates and demotes back to +/-1.
      * Small x stays near erf(x) ~= 2/sqrt(pi) * x.
      * The infinities saturate: erf(+inf)=1, erf(-inf)=-1.
      * NaN propagates: erf(nan)=nan. On the tanh path the underlying HVX tanh
        extern does not propagate NaN, so libdevice.erf guards it explicitly via
        an isnan (x != x) check. On the accurate path the polynomial is odd and
        saturating by construction and propagates NaN natively.
    """

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_erf_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
        USE_TANH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.erf(x, use_tanh_approx=USE_TANH)
        tl.store(output_ptr + offsets, output)

    # Saturation-scale magnitudes, the transition-band edges, near-zero, the
    # infinities, and a NaN. Same probe set as the pytorch-side
    # test_erf_triton_extremes / test_eltwise_erf_fp32_extremes so coverage
    # matches across repos. Padded with in-range zeros so the length stays a
    # power of two for tl.arange in the kernel (a non-power-of-two block would
    # fail to compile).
    vals = [
        -1000.0,
        -100.0,
        -20.0,
        -8.0,
        -4.0,
        -1.0,
        -1e-3,
        0.0,
        1e-3,
        1.0,
        4.0,
        8.0,
        20.0,
        100.0,
        1000.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    vals += [0.0] * (32 - len(vals))
    assert len(vals) == 32

    x = torch.tensor(vals, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    libd_erf_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=x.numel(),
        USE_TANH=use_tanh_approx,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.erf(x)

    # NaN must propagate (erf(nan)=nan); allclose/isfinite don't handle NaN, so
    # check it explicitly and mask it out of the finite/closeness checks.
    nan_mask = torch.isnan(x)
    assert torch.isnan(
        output.float()[nan_mask]
    ).all(), f"erf did not propagate NaN: {output}"
    finite_out = output.float()[~nan_mask]
    # Remaining output must stay finite (no inf/NaN leaking through the poly).
    assert torch.isfinite(finite_out).all(), f"non-finite erf output: {output}"

    assert torch.allclose(
        output[~nan_mask],
        reference[~nan_mask],
        atol=_atol(use_tanh_approx, dtype),
    )
