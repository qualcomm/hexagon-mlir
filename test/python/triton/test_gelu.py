# ===- test_gelu.py ---------------------------------------------------------===
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

# 1e-2 is the minimum ATOL where the fp16 test succeeds
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-2


@triton.jit
def tanh(x):
    result = tl.exp((-2 * x).to(tl.float32)).to(x.dtype)
    result = (1 - result) / (1 + result)
    return result


@pytest.mark.parametrize("num_elements", [16384])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("enable_mt", [True])
def test_gelu(num_elements, num_threads, dtype, enable_mt):
    print(
        f"Running with num-elements: {num_elements}, num-triton-thread: {num_threads}, "
        f"dtype: {dtype}, enable-multi-threading: {enable_mt}"
    )
    print("-" * 100)

    @triton.jit
    @parameterize_func_name(dtype)
    def gelu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = 0.5 * x * (1 + tanh(0.797885 * x + 0.035677 * x * x * x))
        tl.store(output_ptr + offsets, output)

    x = torch.rand(num_elements, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    block_size = num_elements // num_threads
    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    gelu_kernel[grid](
        x,
        output,
        BLOCK_SIZE=block_size,
        enableMultiThreading=true_if_single_threaded and enable_mt,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
    )

    reference = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )

    reference = torch.nn.functional.gelu(x, approximate="tanh")
