# ===- test_online_softmax.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name

triton.runtime.driver.set_active(HexagonDriver())

CONSTANTS = {"NUM_ROWS": 128, "NUM_COLS": 128}

ATOL = 1e-5


# This softmax implementation is based on the paper -- https://arxiv.org/abs/1805.02867
# This algorithm splits the computation into a two-phase approach and this is the implementation
# of softmax which is actually used in flash attention. Hence, this test also serves
# as a good proxy for testing the main functionality of flash attention.
@triton.jit
def online_softmax(
    x_ptr,
    output_ptr,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):

    tid = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, NUM_COLS),
        strides=(NUM_COLS, 1),
        offsets=(tid * BLOCK_ROW_SIZE, 0),
        block_shape=(BLOCK_ROW_SIZE, BLOCK_COL_SIZE),
        order=(1, 0),
    )
    x_block_ptr_alt = x_block_ptr

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS, NUM_COLS),
        strides=(NUM_COLS, 1),
        offsets=(tid * BLOCK_ROW_SIZE, 0),
        block_shape=(BLOCK_ROW_SIZE, BLOCK_COL_SIZE),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_ROW_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_ROW_SIZE], dtype=tl.float32)

    for _ in range(0, NUM_COLS, BLOCK_COL_SIZE):
        x = tl.load(x_block_ptr_alt)
        m_ij = tl.maximum(m_i, tl.max(x, axis=1))
        z = x - m_ij[:, None]
        num = tl.exp(z)
        l_ij = tl.sum(num, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        x_block_ptr_alt = tl.advance(x_block_ptr_alt, (0, BLOCK_COL_SIZE))

    for _ in range(0, NUM_COLS, BLOCK_COL_SIZE):
        x = tl.load(x_block_ptr)
        y = tl.math.exp(x - m_i[:, None]) / l_i[:, None]
        tl.store(output_block_ptr, y)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COL_SIZE))
        output_block_ptr = tl.advance(output_block_ptr, (0, BLOCK_COL_SIZE))


@pytest.mark.parametrize("num_threads", [1, 4])
def test_online_softmax(num_threads):
    x = torch.rand(CONSTANTS["NUM_ROWS"], CONSTANTS["NUM_COLS"])
    output = torch.empty_like(x)

    # num_threads need to be a factor of NUM_ROWS as BLOCK_SIZE can only be a power of 2,
    # due to masking/padding constraints for block_ptrs
    block_row_size = CONSTANTS["NUM_ROWS"] // num_threads

    # Can also be set to any other factor of NUM_COLS other than num_threads,
    # since each thread processes all the columns.
    block_col_size = CONSTANTS["NUM_COLS"] // num_threads

    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    online_softmax[grid](
        x,
        output,
        NUM_ROWS=CONSTANTS["NUM_ROWS"],
        NUM_COLS=CONSTANTS["NUM_COLS"],
        BLOCK_ROW_SIZE=block_row_size,
        BLOCK_COL_SIZE=block_col_size,
        enableMultiThreading=true_if_single_threaded,
        enableVTCMTiling=true_if_single_threaded,
        enableConvertToHexagonmem=true_if_single_threaded,
        enableHexagonmemCopyToDMA=true_if_single_threaded,
    )

    reference = torch.nn.functional.softmax(x, dim=1)
    assert torch.allclose(output, reference, atol=ATOL)
