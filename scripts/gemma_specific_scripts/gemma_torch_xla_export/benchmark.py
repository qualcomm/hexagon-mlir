# ===- benchmark.py --------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Benchmark and validate PyTorch and StableHLO Gemma executions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from inputs import ExportInputs, VariantSpec, make_prefill_inputs


@dataclass
class BenchmarkResult:
    """Timing and validation results for one benchmark variant."""

    mode: str
    use_kv_cache: bool
    baseline_seconds: float
    stablehlo_seconds: float
    tokens: int
    cosine_similarity: float

    @property
    def time_ratio(self) -> float:
        if self.baseline_seconds == 0.0:
            return float("inf")
        return self.stablehlo_seconds / self.baseline_seconds


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if value.is_floating_point():
        summary["finite_count"] = int(torch.isfinite(value).sum().item())
        summary["nan_count"] = int(torch.isnan(value).sum().item())
        summary["inf_count"] = int(torch.isinf(value).sum().item())
    return summary


def cosine_similarity(
    baseline: torch.Tensor,
    converted: torch.Tensor,
) -> float:
    """Compute cosine similarity with stable higher-precision reductions."""
    baseline_flat = baseline.detach().cpu().reshape(-1)
    converted_flat = converted.detach().cpu().reshape(-1)
    if baseline_flat.numel() != converted_flat.numel():
        raise ValueError(
            "Cosine similarity requires equal-sized tensors: "
            f"baseline={baseline_flat.numel()}, "
            f"converted={converted_flat.numel()}"
        )

    dot = 0.0
    baseline_squared_norm = 0.0
    converted_squared_norm = 0.0
    chunk_size = 1 << 20
    for start in range(0, baseline_flat.numel(), chunk_size):
        end = start + chunk_size
        baseline_chunk = baseline_flat[start:end].to(dtype=torch.float64)
        converted_chunk = converted_flat[start:end].to(dtype=torch.float64)
        dot += float(torch.dot(baseline_chunk, converted_chunk).item())
        baseline_squared_norm += float(torch.dot(baseline_chunk, baseline_chunk).item())
        converted_squared_norm += float(
            torch.dot(converted_chunk, converted_chunk).item()
        )

    denominator = math.sqrt(baseline_squared_norm * converted_squared_norm)
    if denominator == 0.0:
        return float("nan")
    # Floating-point reductions can still overshoot the mathematical bound by
    # a few ulps, so enforce the invariant at the reporting boundary.
    return max(-1.0, min(1.0, dot / denominator))


def unpack_cached_output(
    output: Any,
) -> tuple[torch.Tensor, Any]:
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
    ):
        raise TypeError(
            "Expected cached StableHLO output (logits, updated_cache_state)"
        )
    return output


def compare_verified_logits(
    baseline_outputs: list[torch.Tensor],
    converted_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, float]:
    for baseline_output, converted_output in zip(
        baseline_outputs,
        converted_outputs,
    ):
        if not isinstance(converted_output, torch.Tensor):
            raise TypeError(f"Unexpected StableHLO output: {type(converted_output)}")
        if tuple(converted_output.shape) != tuple(baseline_output.shape):
            raise ValueError(
                f"Output shape mismatch: baseline={baseline_output.shape}, "
                f"stablehlo={converted_output.shape}"
            )

    baseline = torch.cat(baseline_outputs, dim=1)
    converted = torch.cat(converted_outputs, dim=1)
    baseline_finite = torch.isfinite(baseline)
    converted_finite = torch.isfinite(converted)
    if not bool(baseline_finite.all() and converted_finite.all()):
        nonfinite_baseline = int((~baseline_finite).sum().item())
        nonfinite_converted = int((~converted_finite).sum().item())
        print(
            "Verification shape passed; non-finite logits found "
            f"numeric comparison (baseline={nonfinite_baseline}, "
            f"stablehlo={nonfinite_converted}).",
            flush=True,
        )
        raise RuntimeError(
            "StableHLO verification produced non-finite output: "
            f"baseline={nonfinite_baseline}, stablehlo={nonfinite_converted}"
        )

    similarity = cosine_similarity(baseline, converted)
    return baseline, similarity


def print_benchmark_tables(
    results: list[BenchmarkResult],
) -> None:
    """Print benchmark timing and validation results as a Markdown table."""

    def table_rows(
        seconds: str,
    ) -> list[str]:
        rows = []
        for result in results:
            kv_cache = "enabled" if result.use_kv_cache else "disabled"
            benchmark_seconds = getattr(result, seconds)
            tps = result.tokens / benchmark_seconds
            rows.append(
                f"| {result.mode} | {kv_cache} | {benchmark_seconds:.17g} | "
                f"{result.tokens} | {tps:.17g} |"
            )
        return rows

    table_header = (
        "| Mode | KV cache | Time (s) | Tokens | " "TPS (Tokens per Second) |"
    )
    table_separator = "|---|---|---:|---:|---:|"
    validation_lines = []
    for result in results:
        kv_cache = "enabled" if result.use_kv_cache else "disabled"
        validation_lines.extend(
            (
                f"Cosine similarity (mode={result.mode}, "
                f"kv_cache={kv_cache}): {result.cosine_similarity:.9f}",
                f"StableHLO/baseline time ratio (mode={result.mode}, "
                f"kv_cache={kv_cache}): {result.time_ratio:.3f}x",
            )
        )

    print(
        "\n".join(
            (
                "Baseline results:",
                table_header,
                table_separator,
                *table_rows("baseline_seconds"),
                "",
                "StableHLO results:",
                table_header,
                table_separator,
                *table_rows("stablehlo_seconds"),
                "",
                *validation_lines,
            )
        ),
        flush=True,
    )


def run_prefill(
    model: Any,
    inputs: tuple[Any, ...],
    use_kv_cache: bool,
) -> list[torch.Tensor]:
    with torch.no_grad():
        output = model(*inputs)
    if use_kv_cache:
        logits, _ = unpack_cached_output(output)
    else:
        logits = output[:, -1:, :]
    return [logits]


def run_uncached_decode(
    model: Any,
    step_inputs: list[tuple[torch.Tensor, ...]],
    context_length: int,
) -> list[torch.Tensor]:
    outputs = []
    with torch.no_grad():
        for step, inputs in enumerate(step_inputs):
            logits = model(*inputs)
            output_position = context_length + step
            outputs.append(logits[:, output_position : output_position + 1, :])
    return outputs


def run_cached_decode(
    model: Any,
    decode_token: torch.Tensor,
    initial_cache_state: tuple[torch.Tensor, ...],
    context_length: int,
    decode_steps: int,
) -> list[torch.Tensor]:
    outputs = []
    cache_state = initial_cache_state
    with torch.no_grad():
        for step in range(decode_steps):
            cache_position = torch.tensor(
                [context_length + step],
                dtype=torch.int64,
            )
            output = model(decode_token, cache_position, cache_state)
            logits, cache_state = unpack_cached_output(output)
            outputs.append(logits)
    return outputs


def timed_execution(
    function: Any,
) -> tuple[list[torch.Tensor], float]:
    """Warm up once, then time one complete benchmark workload."""
    function()
    start = time.perf_counter()
    outputs = function()
    return outputs, time.perf_counter() - start


def run_comparison(
    args: Any,
    variant: VariantSpec,
    stablehlo_program: Any,
    export_inputs: ExportInputs,
    text_model: torch.nn.Module,
    export_model: torch.nn.Module,
    prefill_model: torch.nn.Module | None,
    print_results: bool = True,
) -> tuple[torch.Tensor, BenchmarkResult]:
    print("Running baseline and StableHLO comparison ...", flush=True)
    if variant.mode == "prefill":
        if variant.use_kv_cache:
            baseline_prefill_inputs = make_prefill_inputs(
                export_inputs.context,
                export_model,
            )
        else:
            baseline_prefill_inputs = export_inputs.context
        baseline_workload = lambda: run_prefill(
            export_model if variant.use_kv_cache else text_model,
            baseline_prefill_inputs,
            variant.use_kv_cache,
        )
        stablehlo_workload = lambda: run_prefill(
            stablehlo_program,
            export_inputs.sample,
            variant.use_kv_cache,
        )
        tokens = args.context_length
    elif variant.use_kv_cache:
        if prefill_model is None:
            raise RuntimeError("Cached decode comparison requires a prefill model")
        with torch.no_grad():
            prefill_output = prefill_model(
                *make_prefill_inputs(
                    export_inputs.context,
                    prefill_model,
                )
            )
        _, initial_cache_state = unpack_cached_output(prefill_output)
        baseline_workload = lambda: run_cached_decode(
            export_model,
            export_inputs.decode_token,
            initial_cache_state,
            args.context_length,
            args.decode_steps,
        )
        stablehlo_workload = lambda: run_cached_decode(
            stablehlo_program,
            export_inputs.decode_token,
            initial_cache_state,
            args.context_length,
            args.decode_steps,
        )
        tokens = args.decode_steps
    else:
        baseline_workload = lambda: run_uncached_decode(
            text_model,
            export_inputs.uncached_decode,
            args.context_length,
        )
        stablehlo_workload = lambda: run_uncached_decode(
            stablehlo_program,
            export_inputs.uncached_decode,
            args.context_length,
        )
        tokens = args.decode_steps

    baseline_outputs, baseline_seconds = timed_execution(baseline_workload)
    stablehlo_outputs, stablehlo_seconds = timed_execution(stablehlo_workload)
    baseline, similarity = compare_verified_logits(
        baseline_outputs,
        stablehlo_outputs,
    )
    result = BenchmarkResult(
        mode=variant.mode,
        use_kv_cache=variant.use_kv_cache,
        baseline_seconds=baseline_seconds,
        stablehlo_seconds=stablehlo_seconds,
        tokens=tokens,
        cosine_similarity=similarity,
    )
    if print_results:
        print_benchmark_tables([result])
    return baseline, result
