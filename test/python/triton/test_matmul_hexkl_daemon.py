# ===- test_matmul_hexkl_daemon.py ------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# End-to-end matmul through the HexKL micro-op HMX daemon path, multi-threaded.
# K spans two 32-blocks, so each output tile takes two mm_f16 jobs (clear_acc
# on the first, write_out on the last) accumulating into one tile — exercising
# the partialInFlight lane-sticking path.  M=128 over NUM_THREADS=4 gives each
# worker one 32-row stripe; enableThreadedDispatch fans the workers across
# threads while the single HMX daemon serializes their dispatched jobs.
#
# Dimensions: A[128,64] @ B[64,64] = C[128,64], f16 inputs, f32 output.
# Grid = 4 threads, each computing a [32,64] output stripe (1 row-tile x 2 col).
# scratch=0: this kernel is matmul-only (no HVX worker compute), so it needs no
# per-thread HVX VTCM budget.

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

M, K, N = 128, 64, 64
NUM_THREADS = 4
BLOCK_M = M // NUM_THREADS  # 32


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
):
    pid = tl.program_id(0)

    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(A_block_ptr)
    b = tl.load(B_block_ptr)
    c = tl.dot(a, b)
    tl.store(C_block_ptr, c.to(C.type.element_ty))


def test_matmul_hexkl_daemon():
    # Random f16 inputs in [0,1): all-positive, so the K=64 accumulation is
    # well-conditioned (no cancellation) and the fp16-rounded result stays
    # within tolerance.  Fixed seed keeps any failure reproducible.
    torch.manual_seed(0)
    A_f32 = torch.rand(M, K)
    B_f32 = torch.rand(K, N)

    A = A_f32.to(torch.float16)
    B = B_f32.to(torch.float16)
    C = torch.zeros(M, N, dtype=torch.float32)

    grid = (NUM_THREADS,)
    matmul_kernel[grid](
        A,
        B,
        C,
        M=M,
        K=K,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_K=K,
        BLOCK_N=N,
        stride_am=K,
        stride_ak=1,
        stride_bk=N,
        stride_bn=1,
        stride_cm=N,
        stride_cn=1,
        enableVectorization=True,
        enableSplitReduceGeneric=True,
        enableHVXInlining=True,
        enableSCFLoopUnroll=True,
        enableMultiThreading=False,
        enableThreadedDispatch=True,
        enableHexKL=True,
        hexKLMode="micro-daemon",
        scratch=0,  # matmul-only: no HVX worker scratch; HMX region auto-reserved
    )

    reference = A.float() @ B.float()
    assert torch.allclose(
        C, reference, atol=1e-2, rtol=1e-2
    ), f"Max diff: {(C - reference).abs().max().item():.6f}"
