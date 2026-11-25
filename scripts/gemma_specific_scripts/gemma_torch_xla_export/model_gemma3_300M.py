# ===- model_gemma3_300M.py ------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""EmbeddingGemma model loader and fixed-shape embedding forward pass."""

# The checkpoint is a Sentence-Transformers pipeline, not just a Gemma3
# backbone: it adds masked mean pooling, two trained dense projections, and
# L2 normalization to the backbone's token embeddings. AutoModel loads
# the backbone only, so we explicitly apply the latter layers in a forward pass.
# This must be done explicitly so the entire path is traceable by the torch.export
# and StableHLO lowering without (avoiding the bulkier Sentence-Transformers pipeline).

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


class Gemma3EmbeddingForward(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, checkpoint_path: Path):
        super().__init__()
        self.backbone = backbone
        self.dense_1 = self._load_dense(checkpoint_path / "2_Dense")
        self.dense_2 = self._load_dense(checkpoint_path / "3_Dense")
        backbone_parameter = next(backbone.parameters())
        self.dense_1.to(
            device=backbone_parameter.device,
            dtype=backbone_parameter.dtype,
        )
        self.dense_2.to(
            device=backbone_parameter.device,
            dtype=backbone_parameter.dtype,
        )

    def _attention_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape
        dtype = next(self.backbone.parameters()).dtype
        device = input_ids.device
        minimum = torch.finfo(dtype).min
        valid_keys = attention_mask.to(device=device, dtype=torch.bool)[
            :, None, None, :
        ]
        mask_shape = (batch_size, 1, sequence_length, sequence_length)
        full_mask = torch.zeros(mask_shape, dtype=dtype, device=device)
        full_mask = full_mask.masked_fill(~valid_keys, minimum)
        positions = torch.arange(sequence_length, device=device)
        sliding_window = self.backbone.config.sliding_window
        in_window = (positions[:, None] - positions[None, :]).abs() < sliding_window
        sliding_mask = full_mask.masked_fill(~in_window[None, None, :, :], minimum)
        return {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }

    @staticmethod
    def _load_dense(directory: Path) -> torch.nn.Linear:
        config = json.loads((directory / "config.json").read_text())
        if config["activation_function"] != "torch.nn.modules.linear.Identity":
            raise ValueError(f"Unsupported dense activation in {directory}")
        linear = torch.nn.Linear(
            config["in_features"],
            config["out_features"],
            bias=config["bias"],
        )
        weights = load_file(str(directory / "model.safetensors"), device="cpu")
        linear.load_state_dict(
            {key.removeprefix("linear."): value for key, value in weights.items()},
            strict=True,
        )
        return linear

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = torch.arange(
            input_ids.shape[1],
            dtype=torch.int64,
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=self._attention_masks(input_ids, attention_mask),
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        token_embeddings = output[0]
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = self.dense_2(self.dense_1(pooled))
        return functional.normalize(embeddings, p=2, dim=-1)


class Model:
    def __init__(
        self,
        checkpoint_path: str,
        max_prefix_len: int | None = None,
        **kwargs: Any,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.max_prefix_len = max_prefix_len
        kwargs.setdefault("low_cpu_mem_usage", False)
        self.model = AutoModel.from_pretrained(self.checkpoint_path, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=True,
            local_files_only=True,
        )
        pooling_config = json.loads(
            (self.checkpoint_path / "1_Pooling" / "config.json").read_text()
        )
        if not pooling_config["pooling_mode_mean_tokens"] or any(
            pooling_config[name]
            for name in (
                "pooling_mode_cls_token",
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
                "pooling_mode_weightedmean_tokens",
                "pooling_mode_lasttoken",
            )
        ):
            raise ValueError("Only EmbeddingGemma mean-token pooling is supported")
        sentence_config = json.loads(
            (self.checkpoint_path / "config_sentence_transformers.json").read_text()
        )
        self.query_prompt = sentence_config.get("prompts", {}).get("query", "")
        self.text_model = Gemma3EmbeddingForward(
            self.model,
            self.checkpoint_path,
        ).eval()

    def preprocess_text_prompt(
        self,
        text: str | Sequence[str],
        padding: bool | str = False,
        truncation: bool | None = None,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> Any:
        if max_length is None:
            max_length = self.max_prefix_len
        if truncation is None:
            truncation = max_length is not None
        prompts = [text] if isinstance(text, str) else list(text)
        prompts = [f"{self.query_prompt}{prompt}" for prompt in prompts]
        return self.tokenizer(
            prompts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
