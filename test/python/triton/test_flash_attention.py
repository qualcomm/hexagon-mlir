# ===- test_flash_attention.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

Z, H, N_CTX, D_HEAD = 1, 1, 1024, 64
BLOCK_N = 64
BLOCK_DMODEL = 64
STAGE = 1
NUM_THREADS = 4

assert N_CTX % NUM_THREADS == 0
BLOCK_M = N_CTX // NUM_THREADS

stride_0 = H * N_CTX * D_HEAD
stride_1 = N_CTX * D_HEAD
stride_2 = D_HEAD
stride_3 = 1

SCALE = 0.5


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    qk_scale,  #
    BLOCK_N: tl.constexpr,  #
    N_CTX: tl.constexpr,
):
    # range of values handled by this stage
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(
            q, tl.trans(k)
        )  # TODO: Explicit transpose to override block ptr tranpose creation failure
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        # p = p.to(tl.float32)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def attention_fwd_kernel(
    pid1,  ## Set to 0
    Q,
    K,
    V,
    sm_scale: tl.constexpr,  #
    Out,  #
    H,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    stride_0: tl.constexpr,  #
    stride_1: tl.constexpr,  #
    stride_2: tl.constexpr,  #
    stride_3: tl.constexpr,  #
):

    pid0 = tl.program_id(0)
    start_m = pid0
    off_hz = pid1
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int32) * stride_0 + off_h.to(tl.int32) * stride_1
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 99999999999999.0  # float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        qk_scale,  #
        BLOCK_N,  #
        N_CTX,  #
    )
    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def test_flash_attention():
    query = torch.rand(Z, H, N_CTX, D_HEAD)
    key = torch.rand(Z, H, N_CTX, D_HEAD)
    value = torch.rand(Z, H, N_CTX, D_HEAD)

    matrix_stride_0 = H * N_CTX * D_HEAD
    matrix_stride_1 = N_CTX * D_HEAD
    matrix_stride_2 = D_HEAD
    matrix_stride_3 = 1

    output = torch.zeros_like(query)

    grid = (NUM_THREADS,)
    attention_fwd_kernel[grid](
        pid1=0,
        Q=query,
        K=key,
        V=value,
        sm_scale=SCALE,
        Out=output,
        H=H,  #
        N_CTX=N_CTX,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        BLOCK_N=BLOCK_N,  #
        STAGE=STAGE,
        stride_0=stride_0,
        stride_1=stride_1,
        stride_2=stride_2,
        stride_3=stride_3,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
    )

    reference = F.scaled_dot_product_attention(query, key, value, scale=SCALE)
    assert torch.allclose(output, reference)
