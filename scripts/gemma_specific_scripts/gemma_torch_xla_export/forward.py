# ===- forward.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Text-only Gemma forward pass wrappers"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import (
    StaticCache,
    StaticLayer,
    StaticSlidingWindowLayer,
)


class StaticScalar:
    """Expose a scalar checkpoint value as a Python value during export."""

    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


class TextOnlyForward(torch.nn.Module):
    """Expose fixed-shape text inference with a tensor-only output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        # Gemma4ForConditionalGeneration contains multimodal placeholder handling
        # even for text-only inputs. Use its already-loaded text stack directly so
        # export does not capture data-dependent image/audio branches.
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.lm_head_out = model.lm_head_out
        self.final_logit_softcapping = (
            model.config.get_text_config().final_logit_softcapping
        )
        if self.final_logit_softcapping is not None:
            self.logit_div = model.logit_div
            self.logit_tanh = model.logit_tanh
            self.logit_mul = model.logit_mul

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        logits = self.lm_head(outputs[0])
        logits = self.lm_head_out(logits)
        if self.final_logit_softcapping is not None:
            logits = self.logit_div(logits, self.final_logit_softcapping)
            logits = self.logit_tanh(logits)
            logits = self.logit_mul(logits, self.final_logit_softcapping)
        return logits


def cache_head_shapes(
    config: Any,
) -> tuple[int | list[int], int | list[int]]:
    """Return KV-cache head counts and dimensions for each cached layer."""
    if hasattr(config, "global_head_dim"):
        cached_layer_types = config.layer_types[: -config.num_kv_shared_layers]
        head_dim = [
            config.global_head_dim if layer == "full_attention" else config.head_dim
            for layer in cached_layer_types
        ]
        num_heads = [
            (
                config.num_global_key_value_heads
                if layer == "full_attention" and config.attention_k_eq_v
                else config.num_key_value_heads
            )
            for layer in cached_layer_types
        ]
        return num_heads, head_dim
    return (
        getattr(config, "num_key_value_heads", config.num_attention_heads),
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
    )


class FunctionalStaticLayer:
    """Functional fixed-size KV cache layer used during export."""

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        cache_position: torch.Tensor,
        max_cache_length: int,
    ):
        self.keys = keys
        self.values = values
        self.cache_position = cache_position
        self.max_cache_length = max_cache_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        kv_length = key_states.shape[-2]
        positions = (
            torch.arange(kv_length, device=self.keys.device) + self.cache_position
        )
        self.keys = self.keys.index_copy(2, positions, key_states)
        self.values = self.values.index_copy(2, positions, value_states)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        del query_length
        return self.max_cache_length, 0

    def get_seq_length(self) -> torch.Tensor:
        return self.cache_position


class FunctionalStaticCache:
    """Minimal Cache interface required by Gemma's text model and masks."""

    def __init__(self, layers: list[FunctionalStaticLayer]):
        self.layers = layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].update(
            key_states,
            value_states,
            *args,
            **kwargs,
        )

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int,
    ) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(query_length)


class CachedTextOnlyForward(torch.nn.Module):
    """Expose cached text inference with an explicit functional KV cache."""

    def __init__(
        self,
        model: TextOnlyForward,
        batch_size: int,
        max_cache_length: int,
    ):
        super().__init__()
        self.language_model = model.language_model
        self.lm_head = model.lm_head
        self.lm_head_out = model.lm_head_out
        self.final_logit_softcapping = model.final_logit_softcapping
        if self.final_logit_softcapping is not None:
            self.logit_div = model.logit_div
            self.logit_tanh = model.logit_tanh
            self.logit_mul = model.logit_mul

        config = self.language_model.config
        static_cache = StaticCache(
            config=config,
            max_cache_len=max_cache_length,
        )
        # Static sliding-window updates contain data-dependent control flow.
        # For a fixed exported decode step, a regular static cache has the same
        # behavior within the configured cache length and is exportable.
        for index, layer in enumerate(static_cache.layers):
            if isinstance(layer, StaticSlidingWindowLayer):
                static_cache.layers[index] = StaticLayer(max_cache_length)

        model_parameter = next(self.language_model.parameters())
        with torch.no_grad():
            activation_dtype = self.language_model.embed_tokens(
                torch.zeros(
                    (1, 1),
                    dtype=torch.int64,
                    device=model_parameter.device,
                )
            ).dtype
        num_heads, head_dim = cache_head_shapes(config)
        static_cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=activation_dtype,
            device=model_parameter.device,
        )
        self.cache_shapes = [
            (tuple(layer.keys.shape), tuple(layer.values.shape))
            for layer in static_cache.layers
        ]
        self.cache_dtype = activation_dtype
        self.cache_device = model_parameter.device
        self.max_cache_length = max_cache_length

    def initial_cache_state(self) -> tuple[torch.Tensor, ...]:
        """Create zero-filled K/V tensors for the first cached step."""
        state = []
        for key_shape, value_shape in self.cache_shapes:
            state.append(
                torch.zeros(
                    key_shape,
                    dtype=self.cache_dtype,
                    device=self.cache_device,
                )
            )
            state.append(
                torch.zeros(
                    value_shape,
                    dtype=self.cache_dtype,
                    device=self.cache_device,
                )
            )
        return tuple(state)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache_state: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        layers = []
        for index, _ in enumerate(self.cache_shapes):
            layers.append(
                FunctionalStaticLayer(
                    keys=cache_state[2 * index],
                    values=cache_state[2 * index + 1],
                    cache_position=cache_position[0:1],
                    max_cache_length=self.max_cache_length,
                )
            )
        cache = FunctionalStaticCache(layers)

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=False,
        )
        logits = self.lm_head(outputs[0])
        logits = self.lm_head_out(logits)
        if self.final_logit_softcapping is not None:
            logits = self.logit_div(logits, self.final_logit_softcapping)
            logits = self.logit_tanh(logits)
            logits = self.logit_mul(logits, self.final_logit_softcapping)
        updated_cache_state = []
        for layer in cache.layers:
            updated_cache_state.extend((layer.keys, layer.values))
        return logits, tuple(updated_cache_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache_state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run one cached decode step."""
        return self.forward_with_cache(input_ids, cache_position, cache_state)


class PrefillTextOnlyForward(CachedTextOnlyForward):
    """Expose prompt prefill with an explicit functional KV cache."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Process the prompt and return final-token logits plus the cache."""
        cache_position = torch.zeros(
            (1,),
            dtype=torch.int64,
            device=input_ids.device,
        )
        logits, updated_cache_state = self.forward_with_cache(
            input_ids,
            cache_position,
            cache_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return logits[:, -1:, :], updated_cache_state


def force_eager_attention(model: torch.nn.Module) -> None:
    """Use finite additive masks during StableHLO export."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def freeze_export_scalars(model: torch.nn.Module) -> None:
    """Remove scalar tensor inputs used as fixed quantization constants."""
    scalar_names = (
        "input_bits",
        "output_bits",
        "k_cache_num_bits",
        "v_cache_num_bits",
    )
    for module in model.modules():
        for name in scalar_names:
            value = getattr(module, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                continue
            if name in module._buffers:
                del module._buffers[name]
            setattr(module, name, StaticScalar(int(value.item())))
