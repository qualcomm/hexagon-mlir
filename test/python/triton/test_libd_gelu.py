# ===- test_libd_gelu.py ----------------------------------------------------===
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
from triton.language.extra import libdevice

triton.runtime.driver.set_active(HexagonDriver())

NUM_THREADS = 4
NUM_ELEMENTS = 8192
BLOCK_SIZE = NUM_ELEMENTS // NUM_THREADS

# 1e-2 is the minimum ATOL where the fp16 test succeeds
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_libd_gelu(dtype):

    @triton.jit
    @parameterize_func_name(dtype)
    def libd_gelu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        # In test_gelu.py this is implemented using a separate triton kernel, we have
        # replaced it with libdevice.tanh
        output = 0.5 * x * (1 + libdevice.tanh(0.797885 * x + 0.035677 * x * x * x))
        tl.store(output_ptr + offsets, output)

    x = torch.rand(NUM_ELEMENTS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_THREADS,)
    libd_gelu_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )
