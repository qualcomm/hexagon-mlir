# ===- test_dsp_masked_scatter_untested.py ---------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


import sys

import torch
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from triton.language.extra import libdevice

import triton

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def _masked_scatter_kernel(
    OUT_PTR,
    IDX_PTR,
    SRC_PTR,
    K,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < K

    idx = tl.load(IDX_PTR + offs, mask=m, other=0)
    idx = idx.to(tl.int32)
    vals = tl.load(SRC_PTR + offs, mask=m, other=0)
    tl.store(OUT_PTR + idx, vals, mask=m)


def _masked_scatter_triton_impl(
    inp: torch.Tensor, mask: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    inp_flat = inp.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1)
    src_flat = src.contiguous().view(-1)

    idx = torch.nonzero(mask_flat, as_tuple=False).flatten()
    K = idx.numel()

    if src_flat.numel() < K:
        raise RuntimeError(
            f"masked_scatter: source has {src_flat.numel()} elements but mask has {K} True elements"
        )
    out_flat = inp_flat.clone()

    BLOCK = 1024
    grid = (1,)
    _masked_scatter_kernel[grid](
        out_flat,
        idx,
        src_flat,
        K,
        BLOCK=BLOCK,
    )
    return out_flat


def masked_scatter_triton_(
    inp: torch.Tensor, mask: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    out = _masked_scatter_triton_impl(inp, mask, src)
    return out


def test_correctness():
    torch.manual_seed(0)
    for shape in [(32, 32)]:
        N = torch.tensor(shape).prod().item()
        inp = torch.randn(*shape, dtype=torch.float32)
        mask = torch.rand(*shape) > 0.5
        src = torch.randn(int(mask.sum().item()) + 2, dtype=torch.float32)

        out_ref = torch.masked_scatter(inp, mask, src)

        mask_flat = mask.contiguous().view(-1)
        idx1 = torch.nonzero(mask_flat, as_tuple=False).flatten()
        K1 = idx1.numel()
        print("Value of K is :", K1)
        out_tt = masked_scatter_triton_(inp, mask, src)
        reshaped_out = out_tt.reshape((32, 32))
        print("Ref output from torch ...")
        print(out_ref)
        print("Actual output from triton ...")
        print(reshaped_out)
        print("Index values (idx1) - these are the positions where mask is True:")
        print(idx1)

        assert torch.allclose(out_ref, reshaped_out, atol=0, rtol=0), "Forward mismatch"

    print("All tests passed.")


test_correctness()
