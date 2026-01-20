# ===- test_rope.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name

triton.runtime.driver.set_active(HexagonDriver())

BATCH_SIZE = 4
SEQ_LEN = 8
N_HEADS = 16
N_FEATURES = 32
THETA = 10000.0

# 1e-3 is the minimum ATOL where the fp16 test succeeds
ATOL = 1e-3

BATCH_SIZE_STRIDE = SEQ_LEN * N_HEADS * N_FEATURES
SEQ_LEN_STRIDE = N_HEADS * N_FEATURES
N_HEADS_STRIDE = N_FEATURES
N_FEATURES_STRIDE = 1


@triton.jit
def tl_pow(x, y):
    return tl.exp(tl.log(x) * y)


@triton.jit
def compute_thetas(
    SEQ_LEN: tl.constexpr, N_FEATURES: tl.constexpr, THETA: tl.constexpr
):
    evens = tl.arange(0, N_FEATURES)
    odds = tl.arange(0, N_FEATURES) - 1
    step_mask = evens % 2 == 0
    steps = tl.where(step_mask, evens, odds)

    freqs = 1.0 / tl_pow(THETA, (steps / N_FEATURES))
    m = tl.arange(0, SEQ_LEN)
    m_freqs = m[:, None] * freqs

    cos_freqs = tl.cos(m_freqs)
    sin_freqs = tl.sin(m_freqs)
    return cos_freqs, sin_freqs


def rope_reference_pytorch(x: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Pure-PyTorch RoPE reference that mirrors the Triton kernel math:
      - Same 'steps' construction (pairing even/odd feature indices)
      - Same frequency scaling with THETA
      - Same rotation rule using neighbors (i, i+1) and (i-1, i)
    Shapes:
      x: (BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES)
    """
    B, L, H, D = x.shape
    device = x.device
    # Recreate 'steps' exactly as in compute_thetas(): [0, 0, 2, 2, 4, 4, ...]
    idx = torch.arange(D, device=device, dtype=torch.float32)
    evens = idx
    odds = idx - 1
    step_mask = evens % 2 == 0
    steps = torch.where(step_mask, evens, odds)

    # Frequency scaling and per-token phase
    freqs = 1.0 / (theta ** (steps / D))
    m = torch.arange(L, device=device, dtype=torch.float32)
    m_freqs = m[:, None] * freqs[None, :]  # (L, D)

    cos = torch.cos(m_freqs).reshape(1, L, 1, D)
    sin = torch.sin(m_freqs).reshape(1, L, 1, D)

    # Pairwise rotation: even uses (i, i+1), odd uses (i-1, i)
    x_shift_fwd = torch.roll(x, shifts=-1, dims=3)  # i+1
    x_shift_bwd = torch.roll(x, shifts=+1, dims=3)  # i-1
    rotary_mask = (torch.arange(D, device=device) % 2 == 0).view(1, 1, 1, D)
    rope = torch.where(
        rotary_mask, x * cos - x_shift_fwd * sin, x_shift_bwd * sin + x * cos
    )
    return rope.to(x.dtype)


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rope(num_threads, dtype):
    @triton.jit
    @parameterize_func_name(dtype)
    def rope_kernel(
        x_ptr,
        output_ptr,
        BATCH_SIZE: tl.constexpr,
        BATCH_SIZE_STRIDE: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        SEQ_LEN_STRIDE: tl.constexpr,
        N_HEADS: tl.constexpr,
        N_HEADS_STRIDE: tl.constexpr,
        N_FEATURES: tl.constexpr,
        N_FEATURES_STRIDE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        THETA: tl.constexpr,
    ):

        pid = tl.program_id(0)
        block_offset = pid * BLOCK_SIZE

        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            strides=(
                BATCH_SIZE_STRIDE,
                SEQ_LEN_STRIDE,
                N_HEADS_STRIDE,
                N_FEATURES_STRIDE,
            ),
            offsets=(block_offset, 0, 0, 0),
            block_shape=(BLOCK_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            order=(3, 2, 1, 0),
        )
        x = tl.load(x_block_ptr)

        # Thetas are precomputed in actual impls., given it just depends on shapes and consts
        # Resulting tensors have shape (SEQ_LEN, N_FEATURES)
        # Must reshape to broadcast/multiply with x
        cos_freqs, sin_freqs = compute_thetas(SEQ_LEN, N_FEATURES, THETA)
        broad_cos = cos_freqs.reshape(1, SEQ_LEN, 1, N_FEATURES)
        broad_sin = sin_freqs.reshape(1, SEQ_LEN, 1, N_FEATURES)

        x_shift_one_fwd_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            strides=(
                BATCH_SIZE_STRIDE,
                SEQ_LEN_STRIDE,
                N_HEADS_STRIDE,
                N_FEATURES_STRIDE,
            ),
            offsets=(block_offset, 0, 0, 1),
            block_shape=(BLOCK_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            order=(3, 2, 1, 0),
        )
        x_shift_one_fwd = tl.load(x_shift_one_fwd_block_ptr)
        x_shift_one_bwd_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            strides=(
                BATCH_SIZE_STRIDE,
                SEQ_LEN_STRIDE,
                N_HEADS_STRIDE,
                N_FEATURES_STRIDE,
            ),
            offsets=(block_offset, 0, 0, -1),
            block_shape=(BLOCK_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            order=(3, 2, 1, 0),
        )
        x_shift_one_bwd = tl.load(x_shift_one_bwd_block_ptr)

        rotary_mask = (
            tl.arange(0, BLOCK_SIZE * SEQ_LEN * N_HEADS * N_FEATURES).reshape(
                BLOCK_SIZE, SEQ_LEN, N_HEADS, N_FEATURES
            )
            % 2
            == 0
        )
        even_case = (x * broad_cos) - (x_shift_one_fwd * broad_sin)
        odd_case = (x_shift_one_bwd * broad_sin) + (x * broad_cos)
        rope_embeddings = tl.where(rotary_mask, even_case, odd_case)

        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            strides=(
                BATCH_SIZE_STRIDE,
                SEQ_LEN_STRIDE,
                N_HEADS_STRIDE,
                N_FEATURES_STRIDE,
            ),
            offsets=(block_offset, 0, 0, 0),
            block_shape=(BLOCK_SIZE, SEQ_LEN, N_HEADS, N_FEATURES),
            order=(3, 2, 1, 0),
        )
        tl.store(output_block_ptr, rope_embeddings.to(x.dtype))

    x = torch.rand(BATCH_SIZE, SEQ_LEN, N_HEADS, N_FEATURES, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    block_size = triton.next_power_of_2(BATCH_SIZE // num_threads)
    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    rope_kernel[grid](
        x,
        output,
        BATCH_SIZE=BATCH_SIZE,
        BATCH_SIZE_STRIDE=BATCH_SIZE_STRIDE,
        SEQ_LEN=SEQ_LEN,
        SEQ_LEN_STRIDE=SEQ_LEN_STRIDE,
        N_HEADS=N_HEADS,
        N_HEADS_STRIDE=N_HEADS_STRIDE,
        N_FEATURES=N_FEATURES,
        N_FEATURES_STRIDE=N_FEATURES_STRIDE,
        BLOCK_SIZE=block_size,
        THETA=THETA,
        enableMultiThreading=true_if_single_threaded,
        enableVTCMTiling=true_if_single_threaded,
        enableConvertToHexagonmem=true_if_single_threaded,
        enableHexagonmemCopyToDMA=true_if_single_threaded,
    )

    reference = rope_reference_pytorch(x, THETA)
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=0)
