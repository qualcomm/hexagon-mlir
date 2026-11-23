# ===- test_gather.py ---------------------------------------------------===
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

triton.runtime.driver.set_active(HexagonDriver())

ROWS, COLS = 4, 8


@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    offs = r[:, None] * COLS + c[None, :]

    src = tl.load(src_ptr + offs)
    idx = tl.load(idx_ptr + offs)

    out = tl.gather(src, idx, axis=1)

    tl.store(out_ptr + offs, out)


def test_gather():
    torch.manual_seed(0)

    src = torch.randn(ROWS, COLS, dtype=torch.float32)
    idx = torch.randint(0, COLS, (ROWS, COLS), dtype=torch.int32)
    out = torch.empty(ROWS, COLS, dtype=torch.float32)

    gather_kernel[(1,)](
        src,
        idx,
        out,
        ROWS=ROWS,
        COLS=COLS,
        enableVectorization=True,
        enableVTCMTiling=True,
        enableDoubleBuffering=True,
        enableMultiThreading=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    reference = torch.gather(src, 1, idx.to(torch.int64))
    assert torch.allclose(out, reference, atol=1e-5, rtol=1e-5)
