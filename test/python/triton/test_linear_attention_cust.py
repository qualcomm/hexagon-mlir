#!/usr/bin/env python3

# ===- test_linear_attention_cust.py ---------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


"""Forward linear-attention kernel, tensor-descriptor variant.

This is the make_tensor_descriptor port of test_linear_attention_cust.py
(which uses tl.make_block_ptr). The measured shape is:
  B=1, T=16, H=16, K=64, V=128, dtype=float16
Block pointers are replaced by on-device tensor descriptors
(tl.make_tensor_descriptor + desc.load / desc.store). The transposed K access
is expressed as a natural contiguous descriptor load followed by tl.trans,
because make_tensor_descriptor requires the last dimension to be contiguous.
"""
import os

import pytest
import torch
import triton
import triton.language as tl


try:
    from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

    triton.runtime.driver.set_active(HexagonDriver())
    _HEXAGON_DRIVER_ACTIVE = True
except Exception:
    _HEXAGON_DRIVER_ACTIVE = False


def _env_flag(name: str, default: int) -> bool:
    return os.environ.get(name, str(default)) == "1"


_HEXAGON_KERNEL_OPTS = (
    dict(
        enableVectorization=_env_flag("AI_COMPILER_HEXAGON_ENABLE_VECTORIZE", 1),
        enableVTCMTiling=_env_flag("AI_COMPILER_HEXAGON_ENABLE_VTCM", 0),
        enableMultiThreading=_env_flag("AI_COMPILER_HEXAGON_ENABLE_MULTITHREADING", 0),
        enableThreadedDispatch=_env_flag(
            "AI_COMPILER_HEXAGON_ENABLE_THREADED_DISPATCH", 1
        ),
        enableDoubleBuffering=_env_flag("AI_COMPILER_HEXAGON_ENABLE_DOUBLEBUF", 0),
        enableConvertToHexagonmem=_env_flag("AI_COMPILER_HEXAGON_ENABLE_HEXAGONMEM", 0),
        enableHexagonmemCopyToDMA=_env_flag("AI_COMPILER_HEXAGON_ENABLE_DMA", 0),
        enableSplitReduction=_env_flag("AI_COMPILER_HEXAGON_ENABLE_SPLITRED", 0),
        fusion=_env_flag("AI_COMPILER_HEXAGON_ENABLE_FUSION", 1),
        fusionDoMultiUse=_env_flag("AI_COMPILER_HEXAGON_ENABLE_FUSION_MULTIUSE", 1),
        htp_kernel_gen=_env_flag("AI_COMPILER_HEXAGON_ENABLE_HTP", 1),
        enableLWP=_env_flag("AI_COMPILER_HEXAGON_ENABLE_LWP", 0),
        # Enable the tensor-descriptor in-bounds fast path in
        # RewriteTensorDescriptorToPointer: fully in-bounds tiles take an
        # unmasked fast-path load, while boundary tiles fall to the masked slow
        # path (stores always stay masked). Exercised by the partial-tile case.
        rewriteDescriptorInboundsFastPath=_env_flag(
            "AI_COMPILER_HEXAGON_REWRITE_DESC_INBOUNDS_FASTPATH", 1
        ),
        num_threads=int(os.environ.get("AI_COMPILER_HEXAGON_NUM_THREADS", "6")),
        scratch=int(os.environ.get("AI_COMPILER_HEXAGON_SCRATCH", "0")),
        iterations=int(os.environ.get("AI_COMPILER_HEXAGON_ITERATIONS", "10")),
    )
    if _HEXAGON_DRIVER_ACTIVE
    else {}
)


BKV_LIST = [64]
NUM_WARPS = [4]
autotune_cache_kwargs = {}
exp = tl.exp


_FWD_AUTOTUNE_CONFIGS = (
    [
        triton.Config(
            {"BV": int(os.environ.get("AI_COMPILER_HEXAGON_BV", "64"))},
            num_warps=4,
            num_stages=2,
        )
    ]
    if _HEXAGON_DRIVER_ACTIVE
    else [
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ]
)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=_FWD_AUTOTUNE_CONFIGS,
    key=["H", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_chunk_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp(b_g)
        b_gk = exp(b_g_last - b_g)
        b_gn = exp(b_g_last)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    v = v + (bos * H + i_h) * V
    o = o + (i_k * all + bos).to(tl.int64) * H * V + i_h * V

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Out-of-bounds elements are zero-padded by the descriptor, so no
        # explicit boundary_check is required here.
        h0_desc = tl.make_tensor_descriptor(
            h0 + i_nh * K * V,
            shape=[K, V],
            strides=[V, 1],
            block_shape=[BK, BV],
        )
        b_h = h0_desc.load([i_k * BK, i_v * BV]).to(tl.float32)

    # Descriptors are loop-invariant (base/shape/strides do not depend on i_t);
    # only the load/store offsets change per chunk, so build them once here.
    q_desc = tl.make_tensor_descriptor(
        q, shape=[T, K], strides=[H * K, 1], block_shape=[BT, BK]
    )
    # k is described in its natural [T, K] contiguous layout (last stride == 1,
    # required by make_tensor_descriptor); the transpose to [BK, BT] is done
    # explicitly in-kernel with tl.trans below.
    k_desc = tl.make_tensor_descriptor(
        k, shape=[T, K], strides=[H * K, 1], block_shape=[BT, BK]
    )
    v_desc = tl.make_tensor_descriptor(
        v, shape=[T, V], strides=[H * V, 1], block_shape=[BT, BV]
    )
    o_desc = tl.make_tensor_descriptor(
        o, shape=[T, V], strides=[H * V, 1], block_shape=[BT, BV]
    )

    for i_t in range(0, NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # [BT, BK]
        b_q = q_desc.load([i_t * BT, i_k * BK])
        b_q = (b_q * scale).to(b_q.dtype)
        # natural [BT, BK] load, transposed to [BK, BT] to match the original
        # p_k orientation
        b_k = tl.trans(k_desc.load([i_t * BT, i_k * BK]))
        # [BT, BV]
        b_v = v_desc.load([i_t * BT, i_v * BV])
        last_idx = min(i_t * BT + BT, T) - 1

        # [BT, BT]
        b_s = tl.dot(b_q, b_k)

        # scalar decay
        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.0)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp(b_g)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
        if USE_G or USE_G_GAMMA:
            b_gs = tl.where(m_s & m_t, exp(b_g[:, None] - b_g[None, :]), 0)
            # [BT, BT]
            b_s *= b_gs
            # [BT, BV]
            b_o = (
                tl.dot(b_s.to(b_q.dtype), b_v)
                + tl.dot(b_q, b_h.to(b_q.dtype)) * b_gq[:, None]
            )
            b_v = (b_v * b_gk[:, None]).to(b_v.dtype)
            b_h *= b_gn
        else:
            # [BT, BT]
            b_s *= m_s & m_t
            # [BT, BV]
            b_o = tl.dot(b_s.to(b_q.dtype), b_v) + tl.dot(b_q, b_h.to(b_q.dtype))

        b_h += tl.dot(b_k, b_v)

        o_desc.store([i_t * BT, i_v * BV], b_o.to(o.dtype.element_ty))

    if STORE_FINAL_STATE:
        ht_desc = tl.make_tensor_descriptor(
            ht + i_nh * K * V,
            shape=[K, V],
            strides=[V, 1],
            block_shape=[BK, BV],
        )
        ht_desc.store([i_k * BK, i_v * BV], b_h.to(ht.dtype.element_ty))


def fused_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    chunk_size: int = 16,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5 if scale is None else scale
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    NK = triton.cdiv(K, BK)
    N = B
    BV = int(os.environ.get("AI_COMPILER_HEXAGON_BV", "64"))

    # Automatically enable boundary_check elimination when the grid is safe:
    # grid = (cdiv(V, BV), NK, N*H) launches exactly the right number of tiles
    # iff V%BV==0, K%BK==0, and T%BT==0. Only then can boundary_check be
    # safely removed (no partial tiles in any dimension).
    grid_is_safe = (V % BV == 0) and (K % BK == 0) and (T % BT == 0)
    kernel_opts = dict(_HEXAGON_KERNEL_OPTS)
    if _HEXAGON_DRIVER_ACTIVE:
        kernel_opts["eliminateSafeBoundaryChecks"] = grid_is_safe

    o = v.new_empty(NK, *v.shape, dtype=torch.float) if NK > 1 else torch.empty_like(v)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None

    # On-device tensor descriptors (tl.make_tensor_descriptor) require a global
    # scratch allocator to be registered on the host before launch.
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NK, N * H)

    fused_chunk_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        o=o,
        h0=initial_state,
        ht=ht,
        cu_seqlens=None,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        **kernel_opts,
    )
    if NK > 1:
        o = o.sum(0).to(v)
    return o, ht


# BT (chunk size) is fixed at 16 for this kernel, so a sequence length T
# exercises full tiles only when T % BT == 0. Two regimes are tested:
#   full_tile    T=16 -> T % BT == 0, every chunk is a full tile.
#   partial_tile T=24 -> T % BT == 8, the last chunk is a boundary (partial)
#                tile. The partial regime is what exercises the tensor-
#                descriptor in-bounds fast path: in-bounds tiles take the
#                unmasked fast-path load, the boundary tile takes the masked
#                slow path, and the boundary store stays masked.
# The full-tile baseline length is taken from the environment (default 16); the
# partial-tile case adds half a chunk so T is no longer a multiple of BT.
_CHUNK_SIZE = 16
_T_BASE = int(os.environ.get("AI_COMPILER_HEXAGON_T", str(_CHUNK_SIZE)))
_SEQ_LENS = [_T_BASE, _T_BASE + _CHUNK_SIZE // 2]
_SEQ_LEN_IDS = ["full_tile", "partial_tile"]


def _run_fused_chunk_fwd(T):
    B = int(os.environ.get("AI_COMPILER_HEXAGON_B", "1"))
    H = int(os.environ.get("AI_COMPILER_HEXAGON_H", "16"))
    K = int(os.environ.get("AI_COMPILER_HEXAGON_K", "64"))
    V = int(os.environ.get("AI_COMPILER_HEXAGON_V", "128"))
    dtype = torch.float16
    torch.manual_seed(42)

    # Allocate tensors with 128-byte alignment so Triton infers
    # tt.divisibility=128 for all pointer args (required for DMA).
    def alloc_aligned(shape, dtype, alignment=128):
        elem_bytes = torch.finfo(dtype).bits // 8
        extra = alignment // elem_bytes
        flat = torch.empty(int(torch.tensor(shape).prod().item()) + extra, dtype=dtype)
        offset = (-flat.data_ptr() % alignment) // elem_bytes
        return flat[offset : offset + int(torch.tensor(shape).prod().item())].reshape(
            shape
        )

    q = alloc_aligned((B, T, H, K), dtype).uniform_(-0.5, 0.5)
    k = alloc_aligned((B, T, H, K), dtype).uniform_(-0.5, 0.5)
    v = alloc_aligned((B, T, H, V), dtype).uniform_(-0.5, 0.5)
    state = alloc_aligned((B, H, K, V), dtype).uniform_(-0.5, 0.5)

    o, ht = fused_chunk_fwd(
        q,
        k,
        v,
        scale=K**-0.5,
        initial_state=state,
        output_final_state=True,
        chunk_size=_CHUNK_SIZE,
    )

    assert o.shape == (
        B,
        T,
        H,
        V,
    ), f"Expected o shape {(B, T, H, V)}, got {tuple(o.shape)}"
    assert ht is not None, "Expected final state output"
    assert ht.shape == (
        B,
        H,
        K,
        V,
    ), f"Expected ht shape {(B, H, K, V)}, got {tuple(ht.shape)}"
    tile_kind = "partial" if T % _CHUNK_SIZE else "full"
    print(
        f"PASSED ({tile_kind} tiles, T={T}): o={tuple(o.shape)}, ht={tuple(ht.shape)}"
    )


@pytest.mark.parametrize("T", _SEQ_LENS, ids=_SEQ_LEN_IDS)
def test_fused_chunk_fwd(T):
    _run_fused_chunk_fwd(T)


def main():
    for T in _SEQ_LENS:
        _run_fused_chunk_fwd(T)


if __name__ == "__main__":
    main()
