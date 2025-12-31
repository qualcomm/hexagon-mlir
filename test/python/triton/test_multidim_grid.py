# ===- test_multidim_grid.py ------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from math import prod

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

GRID = (10, 5, 2)


@triton.jit
def flatten_pid(ptr):
    dim_X = tl.num_programs(0)
    dim_Y = tl.num_programs(1)
    dim_Z = tl.num_programs(2)

    stride_X = dim_Y * dim_Z
    stride_Y = dim_Z
    stride_Z = 1

    pid_X = tl.program_id(0)
    pid_Y = tl.program_id(1)
    pid_Z = tl.program_id(2)

    flat_pid = (pid_X * stride_X) + (pid_Y * stride_Y) + (pid_Z * stride_Z)

    tl.store(ptr + flat_pid, flat_pid)


def test_multidim_grid():

    num_elements = prod(GRID)
    result = torch.zeros(num_elements)
    flatten_pid[GRID](
        result,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.arange(num_elements, dtype=torch.float)
    assert torch.allclose(result, reference)
