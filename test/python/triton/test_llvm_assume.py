# ===- test_llvm_assume.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# tl.assume() lowers to llvm.intr.assume in the Triton IR.  triton-shared-opt
# does not register the LLVM dialect, so these ops caused a parse error:
#   Dialect `llvm' not found for custom op 'llvm.intr.assume'
# They are stripped in ttir_to_ttsharedir before invoking triton-shared-opt.
# On Hexagon the assume is a pure hint with no semantic effect.

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

BLOCK_SIZE = 1024


@triton.jit
def add_with_assume_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    # tl.assume() directly emits llvm.intr.assume in the Triton IR.
    tl.assume(BLOCK_SIZE > 0)
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_llvm_assume():
    x = torch.randn(BLOCK_SIZE, dtype=torch.float32)
    y = torch.randn(BLOCK_SIZE, dtype=torch.float32)
    output = torch.empty(BLOCK_SIZE, dtype=torch.float32)

    add_with_assume_kernel[(1,)](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableDoubleBuffering=False,
    )
    reference = x + y
    assert torch.allclose(output, reference, atol=1e-5)
