# ===- test_libd_isnan.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Regression test for libdevice.isnan: exercises the libdevice.isnan symbol
# directly (as emitted by the torch.compile / inductor path for torch.isnan).
# isnan(x) is implemented as `x != x` (fcmpUNE), so NaN inputs must map to 1
# and every finite value (including +/-inf) must map to 0.

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name
from triton.language.extra import libdevice

triton.runtime.driver.set_active(HexagonDriver())


def _run_isnan_kernel(x_flat: torch.Tensor):
    """Run libdevice.isnan on a flattened tensor and compare against
    torch.isnan."""
    dtype = x_flat.dtype
    block_size = triton.next_power_of_2(len(x_flat))
    x_padded = torch.zeros(block_size, dtype=dtype)
    x_padded[: len(x_flat)] = x_flat
    output = torch.empty(block_size, dtype=torch.uint8)

    @triton.jit
    @parameterize_func_name(dtype)
    def isnan_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        inp = tl.load(x_ptr + offsets)
        tl.store(output_ptr + offsets, libdevice.isnan(inp).to(tl.uint8))

    isnan_kernel[(1,)](
        x_padded,
        output,
        BLOCK_SIZE=block_size,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.isnan(x_padded)
    assert torch.equal(output.view(torch.bool), reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_isnan_small(dtype):
    """Small tensor: mix of finite, +/-inf and NaN values. Only the NaN lanes
    may be flagged; the padding zeros must stay 0."""
    x = torch.tensor(
        [0.0, 1.5, -2.0, float("inf"), -float("inf"), float("nan")], dtype=dtype
    )
    _run_isnan_kernel(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_isnan_multivec(dtype):
    """125-element tensor (5x5x5): exercises the multi-vector path. Scatter a
    few NaNs among random finite values."""
    x = (torch.randn(5, 5, 5) * 10).flatten().to(dtype)
    x[3] = float("nan")
    x[42] = float("nan")
    x[100] = float("inf")
    x[101] = float("-inf")
    _run_isnan_kernel(x)
