# ===- test_softmax.py ------------------------------------------------------===
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

NUM_THREADS = 4
NUM_ROWS = 128
NUM_COLS = 128
BLOCK_SIZE = triton.next_power_of_2(NUM_COLS)

# For torch.allclose() check, default absolute tolerance fails (1e-8)
# only for the torch.float16 test (torch.float32 passes with default).
# This is reasonable- current kernel upcasts to float32 to perform
# many operations (and then downcasts to float16 after each operation).
# 1e-3 is the minimum power of 10 where the check passes for torch.float16.
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-3


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
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
        # For each thread spread the computation such that each thread
        # computes softmax for each row where
        # row_idx_j = pid + j*NUM_THREADS
        # where j is the iteration counter for each thread.
        # Hence actual buffer offset would for start of each row would be,
        # offset_j = (pid + j*NUM_THREADS)*NUM_COLS

        pid_row_offset = tl.program_id(0) * NUM_COLS
        pid_row_stride = tl.num_programs(0) * NUM_COLS
        n_elements = NUM_ROWS * NUM_COLS
        col_offsets = tl.arange(0, BLOCK_SIZE)
        for start_n in range(pid_row_offset, n_elements, pid_row_stride):
            offsets = start_n + col_offsets
            x = tl.load(
                x_ptr + offsets, mask=col_offsets < NUM_COLS, other=-float("inf")
            )
            # tl.max upscales to tl.float32 so using tl.reduce directly.
            z = x - tl.reduce(x, 0, _max)
            num = tl.exp(z.to(tl.float32)).to(x.dtype)
            denom = tl.sum(num, axis=0)
            y = num / denom
            tl.store(output_ptr + offsets, y, mask=col_offsets < NUM_COLS)

    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    softmax_kernel[grid](
        x,
        output,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
    )

    reference = torch.nn.functional.softmax(x, dim=1, dtype=dtype)
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )
