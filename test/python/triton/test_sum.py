# ===- test_sum.py ----------------------------------------------------------===
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

ATOL = 1e-2

torch.manual_seed(42)  # Setting a seed for reproducibility.


@pytest.mark.parametrize("num_elements", [1024 * 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vec_add(num_elements, dtype):
    x = torch.randn(num_elements, dtype=dtype)
    output = torch.empty((1), dtype=dtype)

    @triton.jit
    def sum_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE) + (pid * BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = tl.sum(x)
        tl.store(output_ptr + pid, output)

    sum_kernel[1,](
        x,
        output,
        BLOCK_SIZE=num_elements,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
        enableSplitReduction=True,
    )

    reference = torch.sum(x)
    assert torch.allclose(output, reference, atol=ATOL)
