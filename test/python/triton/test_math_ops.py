# ===- test_math_ops.py -----------------------------------------------------===
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

N = 1024
ATOL = 1e-5


@triton.jit
def cos_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    output = tl.cos(x)
    tl.store(output_ptr + offsets, output)


@triton.jit
def sin_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    output = tl.sin(x)
    tl.store(output_ptr + offsets, output)


def test_math_ops():
    X = torch.linspace(0, 3.14, N, dtype=torch.float32)
    Y = torch.empty_like(X)

    cos_kernel[(1,)](
        X,
        Y,
        BLOCK_SIZE=1024,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.cos(X)
    assert torch.allclose(Y, reference, atol=ATOL)

    sin_kernel[(1,)](
        X,
        Y,
        BLOCK_SIZE=1024,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.sin(X)
    assert torch.allclose(Y, reference, atol=ATOL)
