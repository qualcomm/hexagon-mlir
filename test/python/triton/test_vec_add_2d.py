# ===- test_vec_add_2d.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# Using a large input size to test vtcm tiling when enabled
N_ROWS, N_COLUMNS = 64, 128 * 128


@triton.jit
def add_2d_kernel(
    x_ptr, y_ptr, output_ptr, NUM_ROWS: tl.constexpr, NUM_COLS: tl.constexpr
):

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, NUM_COLS),
        strides=(NUM_COLS, 1),
        offsets=(0, 0),
        block_shape=(NUM_ROWS, NUM_COLS),
        order=(1, 0),
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(NUM_ROWS, NUM_COLS),
        strides=(NUM_COLS, 1),
        offsets=(0, 0),
        block_shape=(NUM_ROWS, NUM_COLS),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS, NUM_COLS),
        strides=(NUM_COLS, 1),
        offsets=(0, 0),
        block_shape=(NUM_ROWS, NUM_COLS),
        order=(1, 0),
    )

    x = tl.load(x_block_ptr)
    y = tl.load(y_block_ptr)
    output = x + y
    tl.store(output_block_ptr, output)


def test_vec_add():
    x = torch.rand(N_ROWS, N_COLUMNS)
    y = torch.rand(N_ROWS, N_COLUMNS)
    output = torch.empty_like(y)

    add_2d_kernel[(1,)](
        x,
        y,
        output,
        N_ROWS,
        N_COLUMNS,
        enableVectorization=True,
        enableVTCMTiling=True,
        enableDoubleBuffering=True,
        enableMultiThreading=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = x + y
    assert torch.allclose(output, reference)
