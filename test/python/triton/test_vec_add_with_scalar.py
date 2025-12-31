# ===- test_vec_add_with_scalar.py ------------------------------------------===
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


@pytest.mark.parametrize("num_elements", [1024 * 256])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vec_add(num_elements, num_threads, dtype):
    x = torch.rand(num_elements, dtype=dtype)
    y = torch.rand(num_elements, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    @triton.jit
    def add_kernel_scalar(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    block_size = triton.next_power_of_2(num_elements // num_threads)

    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    add_kernel_scalar[grid](
        x,
        y,
        output,
        num_elements,
        BLOCK_SIZE=block_size,
        enableMultiThreading=true_if_single_threaded,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=true_if_single_threaded,
        enableHexagonmemCopyToDMA=true_if_single_threaded,
    )

    reference = x + y
    assert torch.allclose(output, reference, atol=ATOL)
