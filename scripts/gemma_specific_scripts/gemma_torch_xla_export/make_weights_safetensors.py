# ===- make_weights_safetensors.py -----------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Pack exported NumPy parameter arrays into a Hexagon safetensors bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def _parameter_names(forward_meta_path: Path) -> list[str]:
    metadata = json.loads(forward_meta_path.read_text())
    names = [
        location["name"]
        for location in metadata["input_locations"]
        if location["type_"] == "parameter"
    ]
    if len(names) != len(set(names)):
        raise ValueError("forward.meta contains duplicate parameter names")
    return names


def make_weights_safetensors(
    data_dir: Path,
    forward_meta_path: Path,
    weights_path: Path,
    weights_metadata_path: Path,
) -> int:
    """Pack the parameter arrays referenced by ``forward.meta``."""
    parameter_names = _parameter_names(forward_meta_path)
    tensors: dict[str, torch.Tensor] = {}

    for name in parameter_names:
        array_path = data_dir / name
        if not array_path.is_file():
            raise FileNotFoundError(f"Missing exported parameter array: {array_path}")
        array = np.load(array_path, allow_pickle=False)
        tensors[name] = torch.from_numpy(np.ascontiguousarray(array))

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(weights_path))

    weights_metadata = {
        "names": parameter_names,
        "kept_var_idx": list(range(len(parameter_names))),
    }
    weights_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    weights_metadata_path.write_text(json.dumps(weights_metadata, indent=2) + "\n")
    return len(parameter_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack output/data arrays selected by output/functions/forward.meta "
            "into weights.safetensors."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("output/data"),
        help="Directory containing exported NumPy arrays (default: output/data).",
    )
    parser.add_argument(
        "--forward-meta",
        type=Path,
        default=Path("output/functions/forward.meta"),
        help=(
            "Export metadata identifying parameter arrays "
            "(default: output/functions/forward.meta)."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights.safetensors"),
        help="Output safetensors path (default: weights.safetensors).",
    )
    parser.add_argument(
        "--weights-metadata",
        type=Path,
        default=Path("weights_metadata.json"),
        help="Output runner metadata path (default: weights_metadata.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = make_weights_safetensors(
        args.data_dir,
        args.forward_meta,
        args.weights,
        args.weights_metadata,
    )
    print(f"Wrote {count} parameter tensors to {args.weights}")
    print(f"Wrote metadata to {args.weights_metadata}")


if __name__ == "__main__":
    main()
