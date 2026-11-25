# ===- test_layer_norm_single_pass.py ---------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Single-pass layernorm using E[x^2] - E[x]^2 variance identity.
# A single tl.reduce with a tuple combine_fn accumulates Sum(x) and Sum(x^2)
# together in one loop over the row instead of two dependent passes.
#
# Requires: triton_shared_multi_result_reduction.patch
# ===------------------------------------------------------------------------===

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name

triton.runtime.driver.set_active(HexagonDriver())

EPSILON = 1e-5
NUM_ROWS = 32
BLOCK_SIZE = 32

ATOL = 1e-5


@triton.jit
def _mean_moments_combine(a_sum, a_sumsq, b_sum, b_sumsq):
    """Combine partial (sum, sum-of-squares) pairs for a single-pass moment
    reduction. Lets tl.reduce accumulate both Sum(x) and Sum(x^2) in one sweep
    over the row."""
    return a_sum + b_sum, a_sumsq + b_sumsq


@pytest.mark.parametrize("num_cols", [256, 8192])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_layer_norm_single_pass(num_cols, num_threads, dtype):
    print(
        f"Running single-pass with num-cols: {num_cols}, "
        f"num-triton-thread: {num_threads}, dtype: {dtype}"
    )
    print("-" * 100)

    @triton.jit
    @parameterize_func_name(dtype)
    def block_layer_norm_fwd_kernel_nowrite(
        x_ptr,
        y_ptr,
        weights_ptr,
        biases_ptr,
        mean_ptr,
        rstd_ptr,
        NUM_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        EPSILON: tl.constexpr,
        NUM_COLS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        programs = tl.num_programs(0)

        col_offset = tl.arange(0, NUM_COLS)
        w = tl.load(weights_ptr + col_offset)[None, :]
        b = tl.load(biases_ptr + col_offset)[None, :]

        row_offset = pid * BLOCK_SIZE
        row_stride = programs * BLOCK_SIZE

        for row in range(row_offset, NUM_ROWS, row_stride):
            x_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(NUM_ROWS, NUM_COLS),
                strides=(NUM_COLS, 1),
                offsets=(row, 0),
                block_shape=(BLOCK_SIZE, NUM_COLS),
                order=(1, 0),
            )
            y_block_ptr = tl.make_block_ptr(
                base=y_ptr,
                shape=(NUM_ROWS, NUM_COLS),
                strides=(NUM_COLS, 1),
                offsets=(row, 0),
                block_shape=(BLOCK_SIZE, NUM_COLS),
                order=(1, 0),
            )

            # Single-pass: one tl.reduce accumulates Sum(x) and Sum(x^2)
            # together in one loop. Moments in fp32 to avoid E[x^2]-E[x]^2
            # cancellation on fp16 input.
            x = tl.load(x_block_ptr)
            xf = x.to(tl.float32)
            _sum, _sumsq = tl.reduce(
                (xf, xf * xf), axis=1, combine_fn=_mean_moments_combine
            )
            # Keep 1D for store, reshape to [BLOCK_SIZE, 1] only for broadcast.
            mean = (_sum / NUM_COLS)[:, None]
            mean_sq = (_sumsq / NUM_COLS)[:, None]
            var = mean_sq - mean * mean
            rstd = tl.rsqrt(var + EPSILON)

            x_hat = (x - mean.to(x.dtype)) * rstd.to(x.dtype)
            y = x_hat * w + b
            tl.store(y_block_ptr, y)

    x = torch.rand(NUM_ROWS, num_cols, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)
    weights = torch.rand(num_cols, dtype=dtype)
    biases = torch.rand(num_cols, dtype=dtype)
    means = torch.empty(NUM_ROWS, dtype=dtype)
    rstds = torch.empty(NUM_ROWS, dtype=dtype)

    grid = (num_threads,)
    block_layer_norm_fwd_kernel_nowrite[grid](
        x,
        output,
        weights,
        biases,
        means,
        rstds,
        NUM_ROWS=NUM_ROWS,
        EPSILON=EPSILON,
        NUM_COLS=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.nn.functional.layer_norm(
        x, (num_cols,), weight=weights, bias=biases, eps=EPSILON
    )
    assert torch.allclose(output, reference, atol=ATOL)
