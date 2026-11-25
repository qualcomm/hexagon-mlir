# ===- test_block_rms_norm_composite.py -------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Block RMS norm kernel using the composite ops pipeline.  Uses block pointers
# to load 2D tiles so that the composite pattern produces a 2D
# composite.rms_norm op with a parallel (row) dimension available for
# multi-threading via FormCompositeVirtualThreadsPass.
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

from . import allclose, parametrize_dtypes

triton.runtime.driver.set_active(HexagonDriver())

EPSILON = 1e-5
NUM_COLS = 512
NUM_ROWS = 1024
BLOCK_SIZE = 256


@triton.jit
def block_rms_norm_kernel(
    x_ptr,
    y_ptr,
    weights_ptr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)

    row_offset = pid * BLOCK_SIZE
    row_stride = tl.num_programs(0) * BLOCK_SIZE

    block = tl.arange(0, NUM_COLS)
    g = tl.load(weights_ptr + block)[None, :]

    for curr_row_offset in range(row_offset, NUM_ROWS, row_stride):
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(NUM_ROWS, NUM_COLS),
            strides=(NUM_COLS, 1),
            offsets=(curr_row_offset, 0),
            block_shape=(BLOCK_SIZE, NUM_COLS),
            order=(1, 0),
        )

        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(NUM_ROWS, NUM_COLS),
            strides=(NUM_COLS, 1),
            offsets=(curr_row_offset, 0),
            block_shape=(BLOCK_SIZE, NUM_COLS),
            order=(1, 0),
        )

        x = tl.load(x_block_ptr)
        mean_squared = tl.sum(x * x, axis=1, keep_dims=True) / NUM_COLS
        rms = tl.sqrt(mean_squared + EPSILON).to(x.dtype)

        y = tl.fdiv(x, rms) * g
        tl.store(y_block_ptr, y)


@parametrize_dtypes(torch.float32)
def test_block_rms_norm_composite(dtype):
    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)
    weights = torch.rand(NUM_COLS, dtype=dtype)

    grid = (1,)

    block_rms_norm_kernel[grid](
        x,
        output,
        weights,
        EPSILON=EPSILON,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        enableCompositeOps=True,
        enableMultiThreading=True,
        enableVectorAccumulatorForReduction=True,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableDoubleBuffering=False,
    )

    reference = torch.nn.functional.rms_norm(
        x, (NUM_COLS,), weight=weights, eps=EPSILON
    )

    assert allclose(output, reference)
