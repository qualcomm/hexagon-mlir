# ===- pipeline.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""High-level model loading, export, and benchmark orchestration."""

from __future__ import annotations

from typing import Any

import torch

import benchmark as benchmark_utils
import model_gemma3_300M as gemma3_model
import options as export_options
import stablehlo_export
import stablehlo_to_linalg
from forward import (
    CachedTextOnlyForward,
    PrefillTextOnlyForward,
    force_eager_attention,
)
from inputs import ExportInputs, VariantSpec, prepare_inputs
from options import Options

ALL_VARIANTS = (
    VariantSpec("prefill", True),
    VariantSpec("prefill", False),
    VariantSpec("decode", True),
    VariantSpec("decode", False),
)


def load_base_models(
    args: Options,
) -> tuple[Any, torch.nn.Module]:
    model_wrapper = gemma3_model.Model(
        checkpoint_path=str(args.checkpoint_path),
        max_prefix_len=args.context_length,
    )
    text_model = model_wrapper.text_model.eval()
    force_eager_attention(text_model)
    return model_wrapper, text_model


def make_variant_models(
    args: Options,
    text_model: torch.nn.Module,
    variant: VariantSpec,
) -> tuple[torch.nn.Module, PrefillTextOnlyForward | None]:
    if args.model == "gemma3_300M":
        if variant.mode != "prefill" or variant.use_kv_cache:
            raise ValueError(
                "Gemma3 300M embedding export supports prefill without KV cache"
            )
        return text_model, None

    if not variant.use_kv_cache:
        return text_model, None

    prefill_model = PrefillTextOnlyForward(
        text_model,
        batch_size=args.batch_size,
        max_cache_length=args.max_cache_length,
    ).eval()
    if variant.mode == "prefill":
        return prefill_model, prefill_model
    return (
        CachedTextOnlyForward(
            text_model,
            batch_size=args.batch_size,
            max_cache_length=args.max_cache_length,
        ).eval(),
        prefill_model,
    )


def load_models(
    args: Options,
    variant: VariantSpec,
) -> tuple[
    Any,
    torch.nn.Module,
    torch.nn.Module,
    PrefillTextOnlyForward | None,
]:
    model_wrapper, text_model = load_base_models(args)
    export_model, prefill_model = make_variant_models(
        args,
        text_model,
        variant,
    )
    return model_wrapper, text_model, export_model, prefill_model


def export_variant(
    args: Options,
    variant: VariantSpec,
    model_wrapper: Any,
    text_model: torch.nn.Module,
    export_model: torch.nn.Module,
    prefill_model: PrefillTextOnlyForward | None,
    print_results: bool,
) -> tuple[ExportInputs, torch.Tensor | None, benchmark_utils.BenchmarkResult | None]:
    export_inputs = prepare_inputs(
        args,
        model_wrapper,
        export_model,
        variant,
    )
    stablehlo_program, export_options = stablehlo_export.export_program(
        args,
        export_model,
        export_inputs.sample,
    )
    baseline, result = None, None
    if args.benchmark:
        baseline, result = benchmark_utils.run_comparison(
            args,
            variant,
            stablehlo_program,
            export_inputs,
            text_model,
            export_model,
            prefill_model,
            print_results=print_results,
        )
    stablehlo_path = stablehlo_export.save_bundle(
        args,
        variant,
        stablehlo_program,
        export_options,
        export_inputs,
        baseline,
    )
    stablehlo_to_linalg.lower_stablehlo_to_linalg(
        stablehlo_path,
        stablehlo_path.with_name("linalg.mlir"),
    )
    return export_inputs, baseline, result


def run_single(args: Options) -> None:
    variant = VariantSpec(args.mode, args.use_kv_cache)
    model_wrapper, text_model, export_model, prefill_model = load_models(
        args,
        variant,
    )
    export_variant(
        args,
        variant,
        model_wrapper,
        text_model,
        export_model,
        prefill_model,
        print_results=True,
    )


def run_benchmark_all(args: Options) -> None:
    model_wrapper, text_model = load_base_models(args)
    cached_prefill_model, _ = make_variant_models(
        args,
        text_model,
        VariantSpec("prefill", True),
    )
    cached_decode_model, _ = make_variant_models(
        args,
        text_model,
        VariantSpec("decode", True),
    )

    results = []
    for variant in ALL_VARIANTS:
        if variant.use_kv_cache:
            export_model = (
                cached_prefill_model
                if variant.mode == "prefill"
                else cached_decode_model
            )
            prefill_model = cached_prefill_model
        else:
            export_model = text_model
            prefill_model = None

        _, _, result = export_variant(
            args,
            variant,
            model_wrapper,
            text_model,
            export_model,
            prefill_model,
            print_results=False,
        )
        if result is None:
            raise RuntimeError("Benchmark-all requires benchmark results")
        results.append(result)

    benchmark_utils.print_benchmark_tables(results)


def run(args: Options) -> None:
    export_options.validate_args(args)
    stablehlo_export.prepare_output_dir(args.output_dir)
    if args.benchmark_all:
        run_benchmark_all(args)
    else:
        run_single(args)
