# ===- options.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Command-line options for the Gemma StableHLO exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

Options = argparse.Namespace
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CHECKPOINTS = {
    "gemma3_300M": REPO_ROOT / "gemma3_300M",
}


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        description="Export fixed-shape Gemma text models to StableHLO and Linalg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CHECKPOINTS),
        default="gemma3_300M",
        help="Model architecture and default checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Checkpoint directory; defaults to the selected model's checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory; defaults to a model-specific directory.",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt used to create the fixed-shape example inputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill"),
        default=None,
        help="Export decode (Gemma4) or prefill (Gemma3 embedding).",
    )
    parser.add_argument(
        "--context-length",
        "--context_length",
        type=int,
        default=128,
        help="Number of prompt tokens used for prefill and decode context.",
    )
    parser.add_argument(
        "--max-cache-length",
        "--max_cache_length",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "KV-cache capacity. Defaults to context-length plus decode-steps "
            "(160 with the defaults above)."
        ),
    )
    parser.add_argument(
        "--decode-steps",
        "--decode_steps",
        type=int,
        default=32,
        help="Number of generated decode tokens to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=1,
        help="Static batch size.",
    )
    parser.add_argument(
        "--use-craft",
        action="store_true",
        help="Override checkpoint CRAFT config and enable CRAFT.",
    )
    parser.add_argument(
        "--export-node-metadata",
        action="store_true",
        help="Emit FX node names as StableHLO locations.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark baseline and StableHLO and compare performance/accuracy.",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help=(
            "Benchmark all mode and KV-cache combinations "
            "(prefill/decode with cache enabled/disabled)."
        ),
    )
    parser.add_argument(
        "--no-kv-cache",
        dest="use_kv_cache",
        action="store_false",
        help="Disable KV caching and export the plain text forward.",
    )
    parser.set_defaults(use_kv_cache=None)
    args = parser.parse_args(argv)
    if args.checkpoint_path is None:
        args.checkpoint_path = MODEL_CHECKPOINTS[args.model]
    if args.output_dir is None:
        args.output_dir = Path(
            "output_gemma3_300M" if args.model == "gemma3_300M" else "output"
        )
    if args.mode is None:
        args.mode = "prefill"
    if args.use_kv_cache is None:
        args.use_kv_cache = False
    if args.benchmark_all:
        args.benchmark = True
    if not hasattr(args, "max_cache_length"):
        args.max_cache_length = args.context_length + args.decode_steps
    return args


def validate_args(args: Options) -> None:
    if (
        args.batch_size < 1
        or args.context_length < 1
        or args.decode_steps < 1
        or args.max_cache_length < 1
    ):
        raise ValueError(
            "batch size, context length, decode steps, and cache length "
            "must be positive"
        )
    if args.use_kv_cache and args.max_cache_length < (
        args.context_length + args.decode_steps
    ):
        raise ValueError(
            "max cache length must be at least context length plus decode steps: "
            f"{args.max_cache_length} < "
            f"{args.context_length + args.decode_steps}"
        )
    if not args.checkpoint_path.is_dir():
        raise FileNotFoundError(args.checkpoint_path)
    if args.model == "gemma3_300M":
        if args.mode != "prefill" or args.use_kv_cache:
            raise ValueError(
                "Gemma3 300M embedding export requires prefill mode without KV cache"
            )
        if args.benchmark or args.benchmark_all:
            raise ValueError("Gemma3 300M benchmark mode is not implemented")
        if args.use_craft:
            raise ValueError("--use-craft is not supported with Gemma3 300M")
        if args.context_length > 2048:
            raise ValueError("Gemma3 300M context length cannot exceed 2048")
