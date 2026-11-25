# ===- test_dsp_exponential.py ----------------------------------------------===
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

from .. import dsp

triton.runtime.driver.set_active(HexagonDriver())

# ===------------------------------------------------------------------------===
# KERNELS
# ===------------------------------------------------------------------------===


@triton.jit
def exponential_kernel(
    output_ptr,
    lambd,
    seed,
    offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    random_vals = tl.rand(seed, offsets + offset)
    exponential_vals = -tl.log(random_vals) / lambd

    tl.store(output_ptr + offsets, exponential_vals, mask=mask)


def exponential_triton(tensor, lambd=1.0):
    n_elements = tensor.numel()
    seed = torch.randint(0, 2**31, (1,)).item()
    offset = 0
    output = tensor.contiguous().view(-1)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    exponential_kernel[grid](
        output,
        lambd,
        seed,
        offset,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return tensor


# ===------------------------------------------------------------------------===
# TESTS
# ===------------------------------------------------------------------------===


@dsp
def test_exponential():
    x = torch.empty(1024, 1024, dtype=torch.float32)
    lambd = 1.0

    output = exponential_triton(x, lambd)

    mean = output.mean().item()
    expected_mean = 1.0 / lambd

    std = output.std().item()
    expected_std = 1.0 / lambd

    mean_error = abs(mean - expected_mean) / expected_mean
    std_error = abs(std - expected_std) / expected_std

    tolerance = 0.1

    assert mean_error < tolerance
    assert std_error < tolerance
