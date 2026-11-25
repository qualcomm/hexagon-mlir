# ===- inputs.py -----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Input construction and workload variants for Gemma StableHLO export."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from forward import CachedTextOnlyForward
from options import Options


@dataclass(frozen=True)
class VariantSpec:
    """One fixed-shape export and benchmark variant."""

    mode: Literal["decode", "prefill"]
    use_kv_cache: bool

    def bundle_dir(self, root: Path, separate_variants: bool = False) -> Path:
        if not separate_variants:
            return root
        cache_name = "kv_cache" if self.use_kv_cache else "no_kv_cache"
        return root / f"{self.mode}_{cache_name}"


@dataclass
class ExportInputs:
    """Inputs shared by export and benchmark workloads."""

    sample: tuple[Any, ...]
    baseline: tuple[torch.Tensor, ...]
    context: tuple[torch.Tensor, ...]
    uncached_decode: list[tuple[torch.Tensor, ...]]
    decode_token: torch.Tensor


def make_prefill_inputs(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: Any,
) -> tuple[torch.Tensor, ...]:
    """Build a prompt prefill call with an empty cache."""
    return (*inputs, model.initial_cache_state())


def make_inputs(
    model: Any,
    prompt: str,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts = [prompt] * batch_size
    encoded = model.preprocess_text_prompt(
        prompts,
        padding=False,
        truncation=True,
        max_length=context_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(dtype=torch.int64).contiguous()
    if input_ids.shape[1] == 0:
        raise ValueError("Prompt tokenization produced an empty input")
    repeats = math.ceil(context_length / input_ids.shape[1])
    input_ids = input_ids.repeat(1, repeats)[:, :context_length].contiguous()
    attention_mask = torch.ones_like(input_ids)
    position_ids = (
        torch.arange(
            context_length,
            dtype=torch.int64,
        )
        .expand(batch_size, -1)
        .contiguous()
    )
    return input_ids, attention_mask, position_ids


def make_cached_inputs(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: CachedTextOnlyForward,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Build a one-token decode export sample with an empty cache."""
    input_ids = inputs[0][:, -1:].contiguous()
    cache_position = torch.zeros((1,), dtype=torch.int64)
    return input_ids, cache_position, model.initial_cache_state()


def make_uncached_decode_inputs(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    decode_steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build fixed-shape full-context inputs for each uncached decode step."""
    context_length = inputs[0].shape[1]
    total_length = context_length + decode_steps
    generated_ids = inputs[0][:, -1:].expand(-1, decode_steps)
    input_ids = torch.cat((inputs[0], generated_ids), dim=1)
    position_ids = torch.arange(
        total_length,
        dtype=torch.int64,
    ).expand(input_ids.shape[0], -1)

    step_inputs = []
    for step in range(decode_steps):
        valid_length = context_length + step + 1
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, :valid_length] = 1
        step_position_ids = position_ids.clone()
        step_position_ids.masked_fill_(attention_mask == 0, 0)
        step_inputs.append(
            (
                input_ids,
                attention_mask,
                step_position_ids.contiguous(),
            )
        )
    return step_inputs


def prepare_inputs(
    args: Options,
    model_wrapper: Any,
    export_model: torch.nn.Module,
    variant: VariantSpec,
) -> ExportInputs:
    context_inputs = make_inputs(
        model_wrapper,
        args.prompt,
        args.batch_size,
        args.context_length,
    )
    if args.model == "gemma3_300M":
        context_inputs = (
            context_inputs[0].to(dtype=torch.int32),
            context_inputs[1].to(dtype=torch.int32),
        )
    uncached_decode_inputs = make_uncached_decode_inputs(
        context_inputs,
        args.decode_steps,
    )
    decode_token = context_inputs[0][:, -1:].contiguous()
    if variant.mode == "prefill":
        if variant.use_kv_cache:
            sample_inputs = make_prefill_inputs(
                context_inputs,
                export_model,
            )
        else:
            sample_inputs = context_inputs
        baseline_inputs = context_inputs
    else:
        if variant.use_kv_cache:
            sample_inputs = make_cached_inputs(context_inputs, export_model)
        else:
            sample_inputs = uncached_decode_inputs[0]
        baseline_inputs = sample_inputs
    cache_tensors = len(sample_inputs[-1]) if variant.use_kv_cache else 0
    print(
        "Inputs:",
        f"mode={variant.mode}, ",
        f"{tuple(sample_inputs[0].shape)} {sample_inputs[0].dtype}, "
        f"context_length={args.context_length}, "
        f"decode_steps={args.decode_steps}, "
        f"max_cache_length={args.max_cache_length}, "
        f"cache_tensors={cache_tensors}",
        flush=True,
    )
    return ExportInputs(
        sample=sample_inputs,
        baseline=baseline_inputs,
        context=context_inputs,
        uncached_decode=uncached_decode_inputs,
        decode_token=decode_token,
    )
