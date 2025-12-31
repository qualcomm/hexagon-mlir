# ===- test_vec_add_multithread.py ------------------------------------------===
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

NUM_ELEMENTS = 131072
NUM_THREADS = 4
BLOCK_SIZE = NUM_ELEMENTS // NUM_THREADS


@triton.jit
def add_kernel_mulithreaded(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_vec_add_multithread():
    x = torch.rand(NUM_ELEMENTS)
    y = torch.rand(NUM_ELEMENTS)
    output = torch.empty_like(x)

    grid = (NUM_THREADS,)
    add_kernel_mulithreaded[grid](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
    )

    reference = x + y
    assert torch.allclose(output, reference)
