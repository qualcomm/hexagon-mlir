# ===- test_rsqrt.py --------------------------------------------------------===
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
from triton.language.extra import libdevice

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def rsqrt_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    output = libdevice.rsqrt(x)
    tl.store(output_ptr + offsets, output)


def test_rsqrt():
    x = torch.rand(1024)
    output = torch.empty_like(x)

    rsqrt_kernel[(1,)](x, output, BLOCK_SIZE=1024)

    reference = torch.rsqrt(x)

    assert torch.allclose(output, reference, atol=1e-5)
