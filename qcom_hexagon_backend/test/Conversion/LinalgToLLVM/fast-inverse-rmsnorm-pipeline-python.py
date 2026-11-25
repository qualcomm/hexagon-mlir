#!/usr/bin/env python3

# ===- fast-inverse-rmsnorm-pipeline-python.py ----------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===----------------------------------------------------------------------===
#
# This script holds the input IR for the lit test
# fast_inverse_rmsnorm_pipeline.mlir (FileCheck directives live in the .mlir
# file in the same folder).
#
# It runs lower-libdevice via the Python binding, then pipes the result
# through linalg-hexagon-opt for convert-math-to-llvm + hexagon-fast-inverse.
# This is done because lower-libdevice is not registered in linalg-hexagon-opt
# (it depends on the Triton tt dialect ecosystem which is intentionally kept
# out of the Hexagon opt tool).
#
# If LowerScalarMathPattern registers math::SqrtOp, lower-libdevice will
# convert math.sqrt to a QHL library call (alloca + store + call + load).
# FastInverse cannot see through that opaque pattern and won't fire →
# arith.divf remains in the output → FileCheck fails.

import subprocess
import shutil
import sys

from triton._C.libtriton import ir, qcom_hexagon_backend


def run(mlir_str):
    """Run lower-libdevice → convert-math-to-llvm → fast-inverse."""
    ctx = ir.context()
    qcom_hexagon_backend.load_dialects(ctx)
    mod = qcom_hexagon_backend.parse_mlir_module_from_str(mlir_str, ctx)
    ok = qcom_hexagon_backend.run_lower_libdevice(mod)
    if not ok:
        print("success=False")
        print(mod)
        return

    opt_path = shutil.which("linalg-hexagon-opt")
    assert opt_path, "linalg-hexagon-opt not found on PATH"

    result = subprocess.run(
        [
            opt_path,
            "-pass-pipeline",
            "builtin.module(convert-math-to-llvm,hexagon-fast-inverse)",
        ],
        input=str(mod),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("success=False")
        print(result.stderr, file=sys.stderr)
        return

    print("success=True")
    print(result.stdout)


def main():
    # Scalar divf(1.0, math.sqrt(x)) → Quake_rsqrt(x)
    # This is the rmsnorm pattern: 1.0 / sqrt(variance + eps)
    run(
        r"""
module {
  func.func @scalar_div_one_by_sqrt(%arg0: f32) -> f32 {
    %one = arith.constant 1.000000e+00 : f32
    %sqrt = math.sqrt %arg0 : f32
    %div = arith.divf %one, %sqrt : f32
    return %div : f32
  }
}
"""
    )


if __name__ == "__main__":
    main()
