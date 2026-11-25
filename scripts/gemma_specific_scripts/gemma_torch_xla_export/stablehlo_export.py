# ===- stablehlo_export.py -------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""StableHLO export and bundle serialization helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch_xla.stablehlo as stablehlo

import benchmark as benchmark_utils
from inputs import ExportInputs, VariantSpec
from options import Options


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_export_options(
    args: Options,
) -> stablehlo.StableHLOExportOptions:
    return stablehlo.StableHLOExportOptions(
        include_human_readable_text=True,
        save_weights=True,
        # Keep learned parameters in data/ and externalize supported
        # export-time scalar constants into constants/.
        inline_all_constant=False,
        export_weights=True,
        export_node_metadata=args.export_node_metadata,
    )


def export_program(
    args: Options,
    model: torch.nn.Module,
    sample_inputs: tuple[Any, ...],
) -> tuple[Any, stablehlo.StableHLOExportOptions]:
    print("Running torch.export.export ...", flush=True)
    with torch.no_grad():
        exported = torch.export.export(
            model,
            sample_inputs,
            strict=True,
        )
    print("Converting ExportedProgram to StableHLO ...", flush=True)
    options = make_export_options(args)
    stablehlo_program = stablehlo.exported_program_to_stablehlo(exported, options)
    return stablehlo_program, options


def save_bundle(
    args: Options,
    variant: VariantSpec,
    stablehlo_program: Any,
    options: stablehlo.StableHLOExportOptions,
    export_inputs: ExportInputs,
    baseline: torch.Tensor | None,
) -> Path:
    output_dir = variant.bundle_dir(args.output_dir, args.benchmark_all)
    print(f"Saving StableHLO bundle: {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(output_dir, options)
    stablehlo_text = stablehlo_program.get_stablehlo_text()
    if stablehlo_text is None:
        raise RuntimeError("StableHLO export did not produce textual MLIR")
    stablehlo_path = output_dir / "stablehlo.mlir"
    stablehlo_path.write_text(stablehlo_text)

    metadata = {
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "prompt": args.prompt,
        "mode": variant.mode,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "decode_steps": args.decode_steps,
        "use_kv_cache": variant.use_kv_cache,
        "max_cache_length": (args.max_cache_length if variant.use_kv_cache else None),
        "benchmark": args.benchmark,
        "benchmark_all": args.benchmark_all,
        "inputs": [
            benchmark_utils.tensor_summary(value) for value in export_inputs.sample
        ],
        "output": (
            benchmark_utils.tensor_summary(baseline) if baseline is not None else None
        ),
        "attention_implementation": "eager",
        "torch_version": torch.__version__,
        "torch_xla_version": getattr(__import__("torch_xla"), "__version__", "unknown"),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Export complete.", flush=True)
    return stablehlo_path
