# ===- test_silu.py ---------------------------------------------------------===
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

# 1e-2 is the minimum power of 10 which passes for the fp16 variant
ATOL = 1e-2


@pytest.mark.parametrize("num_elements", [16384])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_silu(num_elements, num_threads, dtype):
    print(
        f"Running with num-elements: {num_elements}, "
        f"num-triton-thread: {num_threads}, dtype: {dtype}"
    )
    print("-" * 100)

    @triton.jit
    @parameterize_func_name(dtype)
    def silu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE) + (pid * BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = x / (1 + tl.exp((-x).to(tl.float32)).to(x.dtype))
        tl.store(output_ptr + offsets, output)

    x = torch.rand(num_elements, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    true_if_single_threaded = num_threads == 1
    block_size = triton.next_power_of_2(num_elements // num_threads)
    grid = (num_threads,)
    silu_kernel[grid](
        x,
        output,
        BLOCK_SIZE=block_size,
        enableMultiThreading=true_if_single_threaded,
        enableVTCMTiling=true_if_single_threaded,
        enableConvertToHexagonmem=true_if_single_threaded,
        enableHexagonmemCopyToDMA=true_if_single_threaded,
    )

    reference = torch.nn.functional.silu(x)
    assert torch.allclose(output, reference, atol=ATOL)
