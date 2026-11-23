# ===- test_libd_gelu.py ----------------------------------------------------===
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

# libdevice.gelu has two implementations, selected per-call by the
# use_tanh_approx constexpr (default resolved from HEXAGON_ENABLE_ACCURACY at
# import time). The tests pin the constexpr explicitly so both paths are covered.
#
# tanh path (use_tanh_approx=True): 0.5*x*(1+tanh(0.797885*x + 0.035677*x**3)),
# compared against F.gelu(approximate="tanh"). Runs on the native HVX tanh
# symbol, so fp32 holds a tight tolerance; fp16 needs 1e-2.
# accurate path (use_tanh_approx=False): 0.5*x*(1+erf(x/sqrt2)) via the accurate
# eager erf polynomial, compared against F.gelu(approximate="none"). The erf
# polynomial is fp32-ULP accurate; fp16 is bounded by fp16 rounding.
ATOL_FP32 = 1e-4
ATOL_FP16 = 1e-2

# Extremes test only: gelu(x)=x*Phi(x) amplifies any absolute error in the Phi
# approximation by ~|x|, so large-magnitude probes need an |x|-proportional
# tolerance on top of the flat atol. The accurate path's erf polynomial
# saturates to +/-1.00016 (a ~1.6e-4 residual), which the 0.5*x factor scales
# to ~0.5*|x|*1.6e-4 (~0.08 at |x|=1000). 3e-4 covers that (8e-5*|x|) plus
# on-device qf32 rounding headroom; the tanh path stays far inside it.
REL_TOL_EXTREMES = 3e-4


@pytest.mark.parametrize("use_tanh_approx", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_gelu(dtype, use_tanh_approx):

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_gelu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
        USE_TANH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        # Exercise libdevice.gelu directly (both the tanh-composite perf path
        # and the accurate erf-based path) so the shared-libdevice change is
        # covered where it is implemented.
        output = libdevice.gelu(x, use_tanh_approx=USE_TANH)
        tl.store(output_ptr + offsets, output)

    x = torch.rand(NUM_ELEMENTS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    libd_gelu_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_TANH=use_tanh_approx,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    approximate = "tanh" if use_tanh_approx else "none"
    reference = torch.nn.functional.gelu(x, approximate=approximate)
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )


@pytest.mark.parametrize("use_tanh_approx", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_gelu_extremes(dtype, use_tanh_approx):
    """Saturation, infinity, and NaN behaviour.

    Two things must hold on both paths:
      * Large finite |x| saturates appropriately (gelu is not symmetric like
        tanh: gelu(x) -> x for x >> 0, gelu(x) -> 0 for x << 0).
      * NaN propagates: gelu(nan)=nan. On the tanh path the underlying HVX tanh
        extern does not propagate NaN, so libdevice.gelu guards it explicitly via
        an isnan (x != x) check. On the accurate path the erf polynomial is odd
        and saturating by construction and propagates NaN natively.

    The literal infinities are deliberately NOT checked for a specific value:
    gelu(x) = 0.5*x*(1+f(x)) is an inf*0 indeterminate form at x=-inf (f(x)
    saturates to -1) and even CPU F.gelu produces nan there (both
    approximate="none" and "tanh"); approximate="none" also returns nan at
    x=+inf. Since the reference itself is not well-defined at the infinities,
    this test only exercises large finite magnitudes plus NaN.

    Tolerance is |x|-proportional, not a flat atol. gelu(x)=x*Phi(x), so an
    absolute error delta in the Phi approximation produces an output error of
    ~|x|*delta -- the error scales with the input magnitude. On the accurate
    path this is significant: the erf polynomial saturates to +/-1.00016 (a
    ~1.6e-4 residual), which gelu's 0.5*x factor amplifies to ~0.5*|x|*1.6e-4
    (e.g. ~0.08 at |x|=1000). A plain rtol cannot capture this because the
    negative tail has reference ~0, so the bound is atol + REL_TOL*|x|. The
    tanh path saturates tanh to exactly +/-1, so 1+tanh=0 cleanly and it stays
    well inside the same bound.
    """

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_gelu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
        USE_TANH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = libdevice.gelu(x, use_tanh_approx=USE_TANH)
        tl.store(output_ptr + offsets, output)

    # Saturation-scale magnitudes, the transition band, and a NaN. Literal
    # infinities are excluded (see docstring: gelu is not well-defined there,
    # even on CPU). Padded with in-range zeros so the length stays a power of
    # two for tl.arange in the kernel.
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
        float("nan"),
    ]
    vals += [0.0] * (32 - len(vals))
    assert len(vals) == 32

    x = torch.tensor(vals, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    libd_gelu_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=x.numel(),
        USE_TANH=use_tanh_approx,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    approximate = "tanh" if use_tanh_approx else "none"
    reference = torch.nn.functional.gelu(x, approximate=approximate)

    # NaN must propagate (gelu(nan)=nan); allclose/isfinite don't handle NaN, so
    # check it explicitly and mask it out of the finite/closeness checks.
    nan_mask = torch.isnan(x)
    assert torch.isnan(
        output.float()[nan_mask]
    ).all(), f"gelu did not propagate NaN: {output}"

    # Remaining output must stay finite (no inf/NaN leaking through the poly
    # on the finite inputs actually exercised here).
    finite_out = output.float()[~nan_mask]
    assert torch.isfinite(finite_out).all(), f"non-finite gelu output: {output}"

    # |x|-proportional tolerance (see docstring): bound = atol + REL_TOL*|x|.
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    out_f = output.float()[~nan_mask]
    ref_f = reference.float()[~nan_mask]
    bound = atol + REL_TOL_EXTREMES * x.float()[~nan_mask].abs()
    err = (out_f - ref_f).abs()
    assert (err <= bound).all(), (
        "gelu diverged beyond |x|-proportional tolerance; "
        f"max err={err.max().item():.3e}, "
        f"worst ratio err/|x|={(err / x.float()[~nan_mask].abs().clamp(min=1.0)).max().item():.3e}"
    )
