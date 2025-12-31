# ===- test_layer_norm.py ---------------------------------------------------===
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
NUM_ROWS = 27

# For torch.allclose() check, default absolute tolerance fails (1e-8)
# 1e-6 is the minimum power of 10 where the check passes.
ATOL = 1e-5


@pytest.mark.parametrize("num_cols", [8192])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float32]
)  # torch.float16 sum reductions don't compile at the moment
def test_layer_norm(num_cols, num_threads, dtype):
    print(
        f"Running with num-cols: {num_cols}, "
        f"num-triton-thread: {num_threads}, dtype: {dtype}"
    )
    print("-" * 100)

    @triton.jit
    @parameterize_func_name(dtype)
    def layer_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        weights_ptr,
        biases_ptr,
        mean_ptr,
        rstd_ptr,
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

            # Compute mean of row
            x = tl.load(x_ptr_incr + block, mask=mask, other=0.0)  # .to(tl.float32)
            mean = tl.sum(x, axis=0) / NUM_COLS

            # Compute variance of row
            x = tl.where(block < NUM_COLS, x - mean, 0.0)
            _var = x * x
            var = tl.sum(_var, axis=0) / NUM_COLS
            rstd = 1 / tl.sqrt(var + EPSILON)

            # Write mean and rstd (used by backward pass)
            tl.store(mean_ptr + row, mean)
            tl.store(rstd_ptr + row, rstd)

            # Normalize and apply linear transformation
            w = tl.load(weights_ptr + block, mask=mask, other=0.0)
            b = tl.load(biases_ptr + block, mask=mask, other=0.0)
            x_hat = x * rstd
            y = x_hat * w + b
            tl.store(y_ptr_incr + block, y, mask=mask)

    x = torch.rand(NUM_ROWS, num_cols, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)
    weights = torch.rand(num_cols, dtype=dtype)
    biases = torch.rand(num_cols, dtype=dtype)
    means = torch.empty(NUM_ROWS, dtype=dtype)
    rstds = torch.empty(NUM_ROWS, dtype=dtype)

    block_size = triton.next_power_of_2(num_cols)
    is_single_threaded = num_threads == 1
    grid = (num_threads,)
    layer_norm_fwd_kernel[grid](
        x,
        output,
        weights,
        biases,
        means,
        rstds,
        EPSILON=EPSILON,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=num_cols,
        BLOCK_SIZE=block_size,
        enableMultiThreading=is_single_threaded,
        enableVTCMTiling=is_single_threaded,
        enableConvertToHexagonmem=is_single_threaded,
        enableHexagonmemCopyToDMA=is_single_threaded,
    )

    reference = torch.nn.functional.layer_norm(
        x, (num_cols,), weight=weights, bias=biases, eps=EPSILON
    )

    assert torch.allclose(output, reference, atol=ATOL)
