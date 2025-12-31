# ===- test_rms_norm.py -----------------------------------------------------===
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

EPSILON = 1e-5
NUM_ROWS = 127
NUM_COLS = 513
BLOCK_SIZE = triton.next_power_of_2(NUM_COLS)

# For torch.allclose() check, default absolute tolerance fails (1e-8)
# 1e-2 is the minimum power of 10 where the check passes.
ATOL = 1e-2


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32]
)  # torch.float16 sum reductions don't compile at the moment
def test_rms_norm(num_threads, dtype):
    @triton.jit
    @parameterize_func_name(dtype)
    def rms_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        weights_ptr,
        BLOCK_SIZE: tl.constexpr,
        EPSILON: tl.constexpr,
        NUM_COLS: tl.constexpr,
        NUM_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        programs = tl.num_programs(0)
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < NUM_COLS
        for row in range(pid, NUM_ROWS, programs):
            # Increments x/y ptr to the current row
            x_ptr_incr = x_ptr + row * NUM_COLS
            y_ptr_incr = y_ptr + row * NUM_COLS

            # Compute RMS of row
            x = tl.load(x_ptr_incr + block, mask=mask, other=0.0)  # .to(tl.float32)
            mean_squared = tl.sum(x * x, axis=0) / NUM_COLS
            rms = tl.sqrt(mean_squared + EPSILON)

            # Normalize and multiply by weights (no bias for rms)
            g = tl.load(weights_ptr + block, mask=mask, other=0.0)
            y = (x / rms) * g
            tl.store(y_ptr_incr + block, y, mask=mask)

    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)
    weights = torch.rand(NUM_COLS, dtype=dtype)

    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    rms_norm_fwd_kernel[grid](
        x,
        output,
        weights,
        EPSILON=EPSILON,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=true_if_single_threaded,
        enableVTCMTiling=true_if_single_threaded,
        enableConvertToHexagonmem=true_if_single_threaded,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.nn.functional.rms_norm(
        x, (NUM_COLS,), weight=weights, eps=EPSILON
    )

    assert torch.allclose(output, reference, atol=ATOL)
