#!/usr/bin/env python3
# ===- triton-pass-opt.py - Minimal Triton MLIR pass runner for lit tests --===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# ===------------------------------------------------------------------------===
#
# Usage: python3 triton-pass-opt.py <pass[,pass,...]> <input.mlir>
#
# Parses Triton MLIR, runs the specified hexagon-backend pass(es), and prints
# the result to stdout. Intended for use in lit tests.
#
# Supported passes:
#   hexagon-eliminate-safe-boundary-checks
#   hexagon-rewrite-masked-access-to-inbounds-fast-path
#   canonicalize
#   cse
# ===------------------------------------------------------------------------===

import os
import sys

os.environ.setdefault("HEXAGON_ARCH_VERSION", "79")

from triton._C.libtriton import ir, passes, qcom_hexagon_backend  # type: ignore


def run(pass_names: list, input_path: str) -> None:
    ctx = ir.context()
    qcom_hexagon_backend.load_dialects(ctx)

    mod = qcom_hexagon_backend.parse_mlir_module_from_file(input_path, ctx)

    pm = ir.pass_manager(ctx)

    for p in pass_names:
        if p == "hexagon-eliminate-safe-boundary-checks":
            qcom_hexagon_backend.add_eliminate_safe_boundary_checks(pm)
        elif p == "hexagon-rewrite-masked-access-to-inbounds-fast-path":
            qcom_hexagon_backend.add_rewrite_masked_access_to_inbounds_fast_path(pm)
        elif p == "canonicalize":
            passes.common.add_canonicalizer(pm)
        elif p == "cse":
            passes.common.add_cse(pm)
        else:
            print(f"Unknown pass: {p}", file=sys.stderr)
            sys.exit(1)

    pm.run(mod, "triton-pass-opt")
    print(str(mod))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: triton-pass-opt.py <pass[,pass,...]> <input.mlir>",
            file=sys.stderr,
        )
        sys.exit(1)
    run(sys.argv[1].split(","), sys.argv[2])
