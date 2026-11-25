# ===- test_inbounds_fast_path_vadd.py --------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Tests for RewriteMaskedAccessToInboundsFastPath.

1-D path: a pointwise vector-add emits inductor's single-sided boundary mask
(`mask = offsets < n_elements`) with masked load/store. When the element count
is NOT a multiple of the block size the last program owns a partial (remainder)
tile. With rewriteMaskedAccessToInboundsFastPath enabled, interior "perfect"
tiles take an unmasked fast path via an scf.if and only the remainder tile
keeps the masked slow path — mirroring the handwritten (eager) DSP kernels.

2-D path: a 2-D pointwise kernel emits a row mask AND a column mask:
  xmask = xindex[:, None] < xnumel
  ymask = yindex[None, :] < ynumel
  mask  = xmask & ymask
The same fast-path strategy applies per dimension: loads are unmasked; stores
are guarded by scf.if((xoff + XBLOCK - 1 < xnumel) AND (yoff + YBLOCK - 1 < ynumel)).

The correctness bar: results must match the eager reference for both the
block-multiple and remainder cases, with the fast path on and off.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def exp_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    """Compute-intensive pointwise kernel: 8-step polynomial chain on x and y.

    output = ((x*x + y) * (x + y*y) + x*y) * ((x - y) * (x + y) + x*y) +
             ((x*x - y*y) * (x*y + 1.0))

    All arithmetic — no transcendentals, no tolerance fudge.  The chain is long
    enough to make this compute-bound rather than memory-bandwidth-bound, so
    removing the per-lane mask is measurable.  The reference is the same
    expression evaluated in PyTorch eager for an exact bitwise-equivalent check.
    """
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    xx = x * x
    yy = y * y
    xy = x * y
    a = xx + y
    b = x + yy
    c = x - y
    d = x + y
    e = a * b + xy
    f = c * d + xy
    g = (xx - yy) * (xy + 1.0)
    output = e * f + g
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize(
    "fast_path", [False, True], ids=["baseline", "inbounds-fast-path"]
)
@pytest.mark.parametrize(
    # block_size = 256. N=1024 is an exact multiple (4 perfect tiles); N=1000
    # leaves a 232-element remainder tile in the last program.
    "N",
    [1024, 1000],
    ids=["exact-multiple", "remainder"],
)
def test_inbounds_fast_path_vadd(N, fast_path):
    block_size = 256
    x = torch.rand(N)
    y = torch.rand(N)
    output = torch.empty_like(x)

    grid = (triton.cdiv(N, block_size),)
    add_kernel[grid](
        x,
        y,
        output,
        N,
        block_size=block_size,
        rewriteMaskedAccessToInboundsFastPath=fast_path,
    )
    reference = x + y
    assert torch.allclose(output, reference)


# ---------------------------------------------------------------------------
# 2-D kernel
# ---------------------------------------------------------------------------


@triton.jit
def add_kernel_2d(
    x_ptr,
    y_ptr,
    output_ptr,
    x_numel,
    y_numel,
    y_stride,
    xblock: tl.constexpr,
    yblock: tl.constexpr,
):
    """2-D pointwise add with the inductor-style row/col AND mask.

    Grid: (cdiv(x_numel, xblock), cdiv(y_numel, yblock)).
    Each program handles an [xblock, yblock] tile.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    xoff = pid_x * xblock
    yoff = pid_y * yblock
    xidx = xoff + tl.arange(0, xblock)
    yidx = yoff + tl.arange(0, yblock)
    # Row/col masks — inductor andi idiom
    xmask = xidx[:, None] < x_numel
    ymask = yidx[None, :] < y_numel
    mask = xmask & ymask
    # Linear offset in the flat [x_numel * y_numel] buffer
    offs = xidx[:, None] * y_stride + yidx[None, :]
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(output_ptr + offs, x + y, mask=mask)


@pytest.mark.parametrize(
    "fast_path", [False, True], ids=["baseline", "inbounds-fast-path"]
)
@pytest.mark.parametrize(
    # (X, Y) pairs: (16, 32) is an exact multiple for xblock=16 / yblock=32;
    # (20, 40) leaves remainder tiles in both dimensions.
    "shape",
    [(16, 32), (20, 40)],
    ids=["exact-multiple", "remainder"],
)
def test_inbounds_fast_path_vadd_2d(shape, fast_path):
    xblock, yblock = 16, 32
    X, Y = shape
    x = torch.rand(X, Y)
    y = torch.rand(X, Y)
    output = torch.empty_like(x)

    grid = (triton.cdiv(X, xblock), triton.cdiv(Y, yblock))
    add_kernel_2d[grid](
        x,
        y,
        output,
        X,
        Y,
        Y,  # y_stride = Y (row-major, contiguous)
        xblock=xblock,
        yblock=yblock,
        rewriteMaskedAccessToInboundsFastPath=fast_path,
    )
    reference = x + y
    assert torch.allclose(output, reference)


