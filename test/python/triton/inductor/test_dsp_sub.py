# ===- test_dsp_sub.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest
import torch
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

import triton

from .. import dsp

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def mixed_precision_sub_kernel(
    x_ptr,  # pointer to input tensor X
    y_ptr,  # pointer to input tensor Y
    out_ptr,  # pointer to output tensor
    n_elements,  # total number of elements
    # dtype flags (constexpr so Triton can specialize the kernel)
    X_IS_FP16: tl.constexpr,
    X_IS_INT32: tl.constexpr,
    Y_IS_FP16: tl.constexpr,
    Y_IS_INT32: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_INT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles a contiguous BLOCK_SIZE chunk of the flattened tensor.
    fp16 and int32 inputs are upcast to fp32 before subtraction.
    int32 - int32 stays in int32. Result is cast to the requested output dtype.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ── load X
    if X_IS_FP16:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    elif X_IS_INT32:
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # fp32

    # ── load Y
    if Y_IS_FP16:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    elif Y_IS_INT32:
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.int32)
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)  # fp32

    # ── compute (fp32 accumulation, or int32 if both inputs were int32)
    result = x - y

    # ── store with requested output dtype
    if OUT_IS_FP16:
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
    elif OUT_IS_INT32:
        tl.store(out_ptr + offsets, result.to(tl.int32), mask=mask)
    else:
        tl.store(out_ptr + offsets, result, mask=mask)  # fp32


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (DSP-only)
# ─────────────────────────────────────────────────────────────────────────────
def mixed_precision_sub(
    x: torch.Tensor,
    y: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Element-wise subtraction of two tensors with (possibly) different dtypes.
    DSP-only: both inputs must be on device.type == 'cpu'.

    Parameters
    ----------
    x, y : Input tensors. Supported dtypes: float16, float32, int32.
           Shapes must match (no broadcasting in this kernel).
    out_dtype : Output dtype. Defaults to:
        - int32 if both inputs are int32
        - float32 if any input is float32 or a float/int mix
        - else x.dtype (fp16 × fp16)
    block_size : Triton program block size (tunable).

    Returns
    -------
    torch.Tensor on DSP with shape equal to x.shape.
    """
    assert (
        x.device.type == "cpu" and y.device.type == "cpu"
    ), "Both tensors must be on CPU (device.type == 'cpu')."
    assert x.shape == y.shape, "Shape mismatch (broadcasting not supported)."

    SUPPORTED = {torch.float16, torch.float32, torch.int32}
    assert x.dtype in SUPPORTED, f"Unsupported x dtype: {x.dtype}"
    assert y.dtype in SUPPORTED, f"Unsupported y dtype: {y.dtype}"

    # Choose output dtype
    if out_dtype is None:
        if x.dtype == torch.int32 and y.dtype == torch.int32:
            out_dtype = torch.int32
        elif x.dtype == torch.float32 or y.dtype == torch.float32:
            out_dtype = torch.float32
        elif x.dtype == torch.int32 or y.dtype == torch.int32:
            out_dtype = torch.float32  # int32 mixed with fp16 → fp32
        else:
            out_dtype = x.dtype  # fp16 × fp16

    out = torch.empty_like(x, dtype=out_dtype, device=x.device)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)

    # ── Triton path (assumes your Triton build targets DSP).
    mixed_precision_sub_kernel[grid](
        x,
        y,
        out,
        n_elements,
        X_IS_FP16=(x.dtype == torch.float16),
        X_IS_INT32=(x.dtype == torch.int32),
        Y_IS_FP16=(y.dtype == torch.float16),
        Y_IS_INT32=(y.dtype == torch.int32),
        OUT_IS_FP16=(out_dtype == torch.float16),
        OUT_IS_INT32=(out_dtype == torch.int32),
        BLOCK_SIZE=block_size,
    )

    # ── Pure PyTorch fallback (uncomment if needed):
    # if True:  # set to True to bypass Triton on DSP
    #     xx = x.to(torch.float32) if x.dtype != torch.int32 else x
    #     yy = y.to(torch.float32) if y.dtype != torch.int32 else y
    #     if x.dtype == torch.int32 and y.dtype == torch.int32:
    #         tmp = xx - yy  # int32
    #     else:
    #         tmp = xx.to(torch.float32) - yy.to(torch.float32)  # fp32 accumulate
    #     out.copy_(tmp.to(out_dtype))

    return out


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
@pytest.mark.parametrize(
    "dx,dy,dout",
    [
        pytest.param(torch.float16, torch.float16, torch.float16, id="f16f16f16"),
        pytest.param(torch.float32, torch.float32, torch.float32, id="f32f32f32"),
        pytest.param(torch.float16, torch.float32, torch.float32, id="f16f32f32"),
        pytest.param(torch.int32, torch.int32, torch.int32, id="i32i32i32"),
        pytest.param(torch.int32, torch.float32, torch.float32, id="i32f32f32"),
        pytest.param(torch.int32, torch.float16, torch.float32, id="i32f16f32"),
    ],
)
def test_sub(dx, dy, dout):
    N = 8

    if dx == torch.int32 or dy == torch.int32:
        x = torch.randint(1, 10, (N,), dtype=torch.int32)
        y = torch.randint(1, 10, (N,), dtype=torch.int32)
        if dx != torch.int32:
            x = x.to(dx)
        if dy != torch.int32:
            y = y.to(dy)
    else:
        x = torch.rand(N).to(dx)
        y = torch.rand(N).to(dy)

    output = mixed_precision_sub(x, y, out_dtype=dout)
    reference = (
        (x.to(torch.float32) - y.to(torch.float32)).to(dout)
        if not (dx == dy == torch.int32)
        else (x - y)
    )

    abs_err = (
        (output.to(torch.float32) - reference.to(torch.float32)).abs().max().item()
    )
    assert abs_err < 1e-2
