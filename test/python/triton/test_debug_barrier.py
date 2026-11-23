# ===- test_debug_barrier.py ------------------------------------------------===
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

BLOCK_SIZE = 1024


@triton.jit
def add_with_barrier_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.debug_barrier()
    y = tl.load(y_ptr + offsets)
    tl.debug_barrier()
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_debug_barrier():
    x = torch.randn(BLOCK_SIZE, dtype=torch.float32)
    y = torch.randn(BLOCK_SIZE, dtype=torch.float32)
    output = torch.empty(BLOCK_SIZE, dtype=torch.float32)

    add_with_barrier_kernel[(1,)](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableDoubleBuffering=False,
    )
    reference = x + y
    assert torch.allclose(output, reference, atol=1e-5)
