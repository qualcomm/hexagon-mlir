# ===- test_block_softmax_composite.py --------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Block-softmax kernel using the composite ops pipeline.  The softmax pattern
# is recognized as a single composite.softmax op and lowered to a hand-tuned
# 3-pass vectorized implementation (max, exp+sum, normalize) with 2x unrolling
# targeting HVX-width vectors.
#
# VTCM, DMA, and VectorAccumulatorForReduction are disabled so the composite
# lowering is the sole optimization path for the softmax computation.
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

NUM_COLS = 512
NUM_ROWS = 1024


from . import parametrize_dtypes


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@parametrize_dtypes(torch.float32)
def test_block_softmax_composite(dtype):
    BLOCK_SIZE = 256

    @triton.jit
    def block_softmax_kernel(
        x_ptr,
        output_ptr,
        NUM_ROWS: tl.constexpr,
        NUM_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        row_offset = pid * BLOCK_SIZE
        row_stride = tl.num_programs(0) * BLOCK_SIZE

        for curr_row_offset in range(row_offset, NUM_ROWS, row_stride):
            x_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(NUM_ROWS, NUM_COLS),
                strides=(NUM_COLS, 1),
                offsets=(curr_row_offset, 0),
                block_shape=(BLOCK_SIZE, NUM_COLS),
                order=(0, 1),
            )

            output_block_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(NUM_ROWS, NUM_COLS),
                strides=(NUM_COLS, 1),
                offsets=(curr_row_offset, 0),
                block_shape=(BLOCK_SIZE, NUM_COLS),
                order=(0, 1),
            )

            x = tl.load(x_block_ptr)
            z = x - tl.reduce(x, 1, _max, keep_dims=True)
            num = tl.exp(z.to(tl.float32)).to(x.dtype)
            denom = tl.sum(num, axis=1, keep_dims=True, dtype=x.dtype)
            y = tl.fdiv(num, denom)
            tl.store(output_block_ptr, y)

    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (1,)

    block_softmax_kernel[grid](
        x,
        output,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        enableCompositeOps=True,
        enableMultiThreading=False,
        enableVectorAccumulatorForReduction=True,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableDoubleBuffering=False,
    )

    reference = torch.nn.functional.softmax(x, dim=1, dtype=dtype)
    assert torch.allclose(output, reference, atol=1e-5)
