# ===- test_dsp_isnan.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

import triton

from .. import dsp, parametrize_dtypes

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def isnan_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    isNan = x != x
    tl.store(output_ptr + offsets, isNan.to(tl.uint8))


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
@parametrize_dtypes(torch.float16, torch.float32)
def test_isnan(dtype):
    x = torch.rand(32, dtype=dtype)
    x[0] = 1.0
    x[1] = float("nan")
    x[10] = float("inf")
    x[15] = float("-inf")
    x[20] = float("nan")

    output = torch.empty(32, dtype=torch.uint8)

    isnan_kernel[(1,)](
        x,
        output,
        BLOCK_SIZE=32,
        # TODO: It seems tile size that uses the minimum of the args would need to be changed
        enableVectorization=False,
    )

    reference = torch.isnan(x)

    assert torch.equal(output.view(torch.bool), reference)
