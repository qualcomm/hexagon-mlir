# ===- test_softmax_scf_multithread.py --------------------------------------===
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

triton.runtime.driver.set_active(HexagonDriver())

NUM_PROGRAMS = 1
NUM_ROWS = 4
NUM_COLS = 65536
BLOCK_SIZE = triton.next_power_of_2(NUM_COLS)
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-3


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_softmax(dtype):

    @triton.jit
    @parameterize_func_name(dtype)
    def softmax_kernel(
        x_ptr,
        output_ptr,
        NUM_ROWS: tl.constexpr,
        NUM_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):

        n_elements = NUM_ROWS * NUM_COLS
        col_offsets = tl.arange(0, BLOCK_SIZE)
        for start_n in range(0, n_elements, NUM_COLS):
            offsets = start_n + col_offsets
            x = tl.load(
                x_ptr + offsets, mask=col_offsets < NUM_COLS, other=-float("inf")
            )
            z = x - tl.reduce(x, 0, _max)
            num = tl.exp(z.to(tl.float32)).to(x.dtype)
            denom = tl.sum(num, axis=0)
            y = num / denom
            tl.store(output_ptr + offsets, y, mask=col_offsets < NUM_COLS)

    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_PROGRAMS,)
    softmax_kernel[grid](
        x,
        output,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        enableSCFThreading=True,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
    )

    reference = torch.nn.functional.softmax(x, dim=1, dtype=dtype)
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )
