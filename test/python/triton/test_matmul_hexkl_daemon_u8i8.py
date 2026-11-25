# ===- test_matmul_hexkl_daemon_u8i8.py -------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# End-to-end quantized matmul through the HexKL micro-op HMX daemon path.
# int8 x int8 -> int32, multi-threaded.
#
# HMX uses unsigned activations and signed weights. The lowering computes signed
# dot exactly as mm_u8i8(A ^ 0x80, B) - 128 * colsum(B), so this test is bit-exact.
#
# Four programs cover partial M tiles, K/N tiling, and multiple daemon lanes.

import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

NUM_PROGRAMS = 4

# Per-instance VTCM: 8 activation slots, 8 weight slots, and one i32 accumulator
# (32768 B total).
SCRATCH_BYTES = 8 * 2048 + 8 * 1024 + 8192
# Keep per-worker slices while avoiding excessive single-page VTCM use.
MULTI_LANE_SCRATCH_BYTES = 2048


@triton.jit
def matmul_i8_kernel(
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
    # Signed int8 x int8 accumulating into int32.  Triton requires both operands
    # to share a dtype, so the mixed unsigned/signed contract HMX actually wants
    # cannot be written here -- the backend derives it (see the module header).
    c = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C_block_ptr, c)


@pytest.mark.parametrize(
    "BLOCK_M,K,N,num_threads,scratch_bytes",
    [
        pytest.param(32, 32, 128, 1, SCRATCH_BYTES, id="partial-m-single-k-multi-n"),
        pytest.param(64, 128, 64, 1, SCRATCH_BYTES, id="multi-k"),
        pytest.param(64, 64, 64, 3, MULTI_LANE_SCRATCH_BYTES, id="three-daemon-lanes"),
    ],
)
def test_matmul_hexkl_daemon_u8i8(BLOCK_M, K, N, num_threads, scratch_bytes):
    torch.manual_seed(0)
    M = NUM_PROGRAMS * BLOCK_M

    # Full int8 range, endpoints included: randint's high bound is exclusive, so
    # this yields [-128, 127].  The -128 and +127 values are the ones that break
    # a sloppy bias (0x80 flip) or a missing colsum correction, so they must be
    # present rather than sampled away.
    A = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), dtype=torch.int8)
    C = torch.zeros(M, N, dtype=torch.int32)

    grid = (NUM_PROGRAMS,)
    matmul_i8_kernel[grid](
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
        enableThreadedDispatch=False,
        enableHexKL=True,
        hexKLMode="micro-daemon",
        scratch=scratch_bytes,
        num_threads=num_threads,
    )

    # Integer arithmetic throughout, so the expected result is exact.  Widen to
    # int32 before multiplying so the reference itself cannot overflow (|result|
    # <= K*128*128 = 1048576 here, comfortably inside int32).
    reference = A.to(torch.int32) @ B.to(torch.int32)

    if not torch.equal(C, reference):
        diff = (C - reference).abs()
        bad = int((diff != 0).sum().item())
        idx = torch.nonzero(diff, as_tuple=False)[0].tolist()
        r, c_ = idx
        pytest.fail(
            f"BLOCK_M={BLOCK_M}, K={K}, N={N}, num_threads={num_threads}, "
            f"scratch={scratch_bytes}: "
            f"{bad}/{M * N} elements differ (expected bit-exact). "
            f"First mismatch at [{r},{c_}]: got {int(C[r, c_])}, "
            f"expected {int(reference[r, c_])}, max abs diff "
            f"{int(diff.max().item())}"
        )
