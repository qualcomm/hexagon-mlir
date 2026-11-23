# ===- test_vtcm_staging_skip_perf.py ----------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Perf test for VTCMTiling's pure-elementwise VTCM-staging skip.

VTCMTilingPass now skips VTCM staging (alloc + DMA-copy-in/out) for pure
elementwise linalg.generic ops -- every operand element is touched exactly
once regardless of tiling, so staging only adds overhead.

The `tileSizes` kernel-launch kwarg maps to VTCMTiling's `tile-size-override`
pass option. Left empty (default), the pass takes the new skip-staging path
for this pure elementwise kernel. Set to a non-empty value, it forces the
pre-fix staged behavior via the pass's existing user-override escape hatch --
giving a real before/after A/B on the same kernel.
"""

import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())


@triton.jit
def load_store_kernel(x_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)
