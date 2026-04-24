# ===- test_static_allocator_vadd.py ----------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Tests for the static VTCM allocator (MemoryOffsetsPass + InsertScratchArgPass).

Exercises external VTCM scratch across grid sizes [1, 2, 4] with and without
scratch enabled. When scratch > 0 the compiler auto-configures
enableThreadedDispatch, disables enableConvertToHexagonmem, etc. via
parse_options().
"""

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize(
    "use_scratch", [False, True], ids=["no-scratch", "vtcm-scratch"]
)
@pytest.mark.parametrize("grid", [(1,), (2,), (4,)], ids=["grid1", "grid2", "grid4"])
def test_static_allocator_vadd(grid, use_scratch):
    N = 131072
    x = torch.rand(N)
    y = torch.rand(N)
    output = torch.empty_like(x)

    alignment = 2048
    per_instance_vtcm = (1 * 1024 * 1024) & ~(alignment - 1)
    scratch = per_instance_vtcm if use_scratch else 0

    num_instances = grid[0]
    elements_per_instance = N // num_instances

    add_kernel[grid](
        x,
        y,
        output,
        N,
        block_size=elements_per_instance,
        enableHexagonmemCopyToDMA=False,
        scratch=scratch,
    )
    reference = x + y
    assert torch.allclose(output, reference)
