# ===- weight_prepack.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Host-side pre-pack for resident HMX weights (P2).

The compiler publishes a contract in the module attribute
``hmx.weight_prepack`` (a JSON object, handed to the launcher through the kernel
metadata). It names, for each runtime weight it made resident, the function
argument *slot*, the logical shape and the crouton shape, plus the crouton
permutation as an affine coefficient map.

When the contract is present, the launcher must hand the kernel that argument
already in crouton order: the kernel no longer emits ``hmx.pack_weight`` (it
copies the argument into VTCM once and pins it). This module is the host half of
that contract. It packs the permutation from the compiler's own coefficient map,
and it verifies its vectorised fast path against that map on a small canary, so
a future change to the compiler layout fails loudly instead of silently
corrupting the weight.

Identity/caching: packing is keyed by ``(slot, shape, dtype, content hash)`` --
a content-addressed cache, never a tensor `data_ptr` cache -- so repeated
launches of the same weight reuse one packed image.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np

_TILE = 32


class WeightPrepack:
    """Parsed ``hmx.weight_prepack`` contract plus a content-addressed cache."""

    def __init__(self, layout: Optional[dict], weights: list):
        self._by_slot: Dict[int, dict] = {}
        for w in weights:
            self._by_slot[int(w["slot"])] = w
        self._layout = layout
        self._layout_verified = False
        self._cache: Dict[Tuple, bytes] = {}

    @classmethod
    def from_metadata(
        cls, weight_prepack_json: Optional[str]
    ) -> Optional["WeightPrepack"]:
        """Parse the metadata string; None when the contract is absent/empty."""
        if not weight_prepack_json:
            return None
        try:
            doc = json.loads(weight_prepack_json)
        except (TypeError, ValueError):
            return None
        weights = doc.get("weights") or []
        if not weights:
            return None
        return cls(doc.get("layout"), weights)

    def has_slot(self, slot: int) -> bool:
        return slot in self._by_slot

    def pack(self, tensor, slot: int) -> Optional[bytes]:
        """Return the crouton-ordered bytes for `tensor`, or None if it does not
        match the slot's contract (the caller then falls back to the raw bytes,
        which is correct only if the gate is off)."""
        desc = self._by_slot.get(slot)
        if desc is None:
            return None
        shape = tuple(int(x) for x in desc["shape"])
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if tuple(arr.shape) != shape:
            return None
        if arr.dtype != np.float16:
            return None
        raw = arr.tobytes()
        key = (
            slot,
            shape,
            str(arr.dtype),
            hashlib.blake2b(raw, digest_size=16).hexdigest(),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        packed = self._pack_crouton(arr, shape, desc)
        self._cache[key] = packed
        return packed

    @staticmethod
    def _permute(bits: np.ndarray, t0: int, t1: int, j: int, c: int,
                 h: int) -> np.ndarray:
        """Permute a row-major ``[K, N]`` f16 image into weight crouton order.

        The compiler's weight grid is ``[Nt, Kt, 16, 32, 2]`` with logical
        ``[N, K]``, so ``t0 = Nt`` and ``t1 = Kt``. Flat row-major ``[K, N]``
        reshapes to ``(Kt, j, h, Nt, c)`` -- with ``k = ((kt) * j + jj) * h +
        hh`` and ``n = nt * c + cc`` -- and transposes to crouton order
        ``(Nt, Kt, j, c, h)`` = ``(n_tile, k_tile, j, c, h)``. Shared by the
        fast pack and the canary so the two cannot drift apart.
        """
        mid = bits.reshape(t1, j, h, t0, c)
        return np.ascontiguousarray(mid.transpose(3, 0, 1, 4, 2))

    def _pack_crouton(self, arr: np.ndarray, shape, desc) -> bytes:
        t0, t1, j, c, h = (int(x) for x in desc["crouton"])
        rows, cols = shape
        if rows != t1 * _TILE or cols != t0 * _TILE:
            raise ValueError(
                f"crouton grid {desc['crouton']} does not tile shape {shape}"
            )
        if not self._layout_verified:
            self._verify_layout(t0, t1, j, c, h)
            self._layout_verified = True
        # f16 bits are preserved through the permutation.
        bits = arr.view(np.uint16)
        return self._permute(bits, t0, t1, j, c, h).tobytes()

    def _verify_layout(self, t0, t1, j, c, h) -> None:
        """Check the vectorised pack against the compiler's coefficient map.

        Uses a fixed small canary on both sides, so a mismatch means the host's
        permutation and the compiler's have drifted apart -- refuse to pack
        rather than feed the engine garbage.
        """
        if not self._layout or "results" not in self._layout:
            raise RuntimeError(
                "weight prepack metadata has no layout; refusing to guess the "
                "crouton permutation"
            )
        ct0 = ct1 = 2
        rows, cols = ct1 * _TILE, ct0 * _TILE
        ramp = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
        fast = self._permute(ramp, ct0, ct1, j, c, h)
        results = self._layout["results"]
        slow = np.empty(fast.shape, dtype=np.uint16)
        for phys in np.ndindex(*fast.shape):
            r = sum(coeff * phys[d] for d, coeff in results[0])
            col = sum(coeff * phys[d] for d, coeff in results[1])
            if not (0 <= r < rows and 0 <= col < cols):
                raise RuntimeError(
                    "host crouton pack disagrees with the compiler layout "
                    "map; refusing to pre-pack the weight"
                )
            slow[phys] = ramp[r, col]
        if not np.array_equal(fast, slow):
            raise RuntimeError(
                "host crouton pack disagrees with the compiler layout map; "
                "refusing to pre-pack the weight"
            )
