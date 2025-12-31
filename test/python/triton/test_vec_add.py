# ===- test_vec_add.py ------------------------------------------------------===
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
BLOCK_SIZE = 131072


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_vec_add():
    x = torch.rand(BLOCK_SIZE)
    y = torch.rand(BLOCK_SIZE)
    output = torch.empty_like(x)

    # best performance currently is with just MT on.
    # Rest are enabled for correctness checks.
    add_kernel[(1,)](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    reference = x + y
    assert torch.allclose(output, reference)
