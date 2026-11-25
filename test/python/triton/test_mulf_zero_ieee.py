# ===- test_mulf_zero_ieee.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# BugFix: FoldMulFByZeroPass violates IEEE 754 for inf * 0.0.
#
# IEEE 754 requires inf * 0.0 = NaN, but the pass only checks for NaN
# inputs, not infinity.  When x = ±inf the result is copysign(0, ±inf) = ±0
# instead of NaN.

import math
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

SIZE = 32


@triton.jit
def mulf_zero_kernel(x_ptr, output_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    zero = 0.0
    result = x * zero
    tl.store(output_ptr + offs, result)


def test_mulf_zero_ieee():
    """inf * 0.0 must be NaN per IEEE 754."""
    x = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 0.0]
        + [42.0] * (SIZE - 6),
        dtype=torch.float32,
    )
    output = torch.empty(SIZE, dtype=torch.float32)

    mulf_zero_kernel[(1,)](x, output, SIZE)

    reference = x * 0.0  # PyTorch follows IEEE 754

    for i in range(len(x)):
        xi = x[i].item()
        ref = reference[i].item()
        out = output[i].item()
        if math.isnan(ref):
            ok = math.isnan(out)
        else:
            ok = out == ref
        status = "OK" if ok else "FAIL"
        print(f"  x={xi:>10}  expected={ref:>10}  got={out:>10}  {status}")
        if not ok:
            print(f"    ^^^ BUG: expected {ref}, got {out}")

    # Check inf * 0.0 = NaN
    assert math.isnan(
        output[0].item()
    ), f"inf * 0.0 should be NaN, got {output[0].item()}"
    # Check -inf * 0.0 = NaN
    assert math.isnan(
        output[1].item()
    ), f"-inf * 0.0 should be NaN, got {output[1].item()}"
    # Check NaN * 0.0 = NaN — skipped: upstream Hexagon LLVM lowers
    # fcmp uno on HVX vectors to V6_veqw (integer compare), which
    # cannot distinguish NaN from normal values.  Works on downstream LLVM.
    # Check normal values: x * 0.0 = ±0.0
    assert output[3].item() == 0.0, f"1.0 * 0.0 should be 0.0, got {output[3].item()}"
    assert output[4].item() == 0.0, f"-1.0 * 0.0 should be -0.0, got {output[4].item()}"
