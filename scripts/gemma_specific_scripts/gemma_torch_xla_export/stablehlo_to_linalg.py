# ===- stablehlo_to_linalg.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Lower StableHLO MLIR to Linalg using JAX's MLIR bindings."""

from __future__ import annotations

from pathlib import Path

_PASS_PIPELINE = (
    "builtin.module(" "stablehlo-legalize-to-linalg{enable-primitive-ops=true}" ")"
)


def lower_stablehlo_to_linalg(input_path: Path, output_path: Path) -> Path:
    """Lower ``input_path`` and write the resulting Linalg module."""
    from jax._src.interpreters import mlir as jax_mlir
    from jaxlib.mlir import ir, passmanager
    from jaxlib.mlir._mlir_libs import _stablehlo

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Register all dialects and the StableHLO conversion passes in one context.
    ctx = jax_mlir.make_ir_context()
    _stablehlo.register_stablehlo_passes()
    with ctx:
        print(f"  Parsing {input_path} ...", flush=True)
        module = ir.Module.parse(input_path.read_text())

        print("  Running stablehlo-legalize-to-linalg ...", flush=True)
        pass_manager = passmanager.PassManager.parse(_PASS_PIPELINE, ctx)
        pass_manager.run(module.operation)

        print(f"  Writing {output_path} ...", flush=True)
        output_path.write_text(str(module))

    print("Done.", flush=True)
    return output_path
