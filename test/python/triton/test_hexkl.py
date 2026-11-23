# ===- test_hexkl.py -------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

BATCH_SIZE = 8
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
N = 256
HALF_FLOAT = True


@triton.jit
def matmul_kernel(
    a_ptr,
    stride_ab,
    stride_an,
    stride_ah,
    b_ptr,
    stride_bb,
    stride_bh,
    stride_bi,
    c_ptr,
    stride_cb,
    stride_cn,
    stride_ci,
    BATCH_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Computes C = AB in one tl.dot()

    where A has shape [N,           HIDDEN_SIZE]
          B has shape [HIDDEN_SIZE, INTERMEDIATE_SIZE]

    BLOCK_SIZE_I and BLOCK_SIZE_N are two tiled dimensions

    """

    pid = tl.program_id(axis=0)

    n_i = tl.cdiv(INTERMEDIATE_SIZE, BLOCK_SIZE_I)

    pid_i = pid % n_i
    pid_n = pid // n_i

    start_i = pid_i * BLOCK_SIZE_I
    start_n = pid_n * BLOCK_SIZE_N

    idx_b = tl.arange(0, BATCH_SIZE)
    mask_b = idx_b == idx_b

    idx_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = idx_n < N

    idx_i = start_i + tl.arange(0, BLOCK_SIZE_I)
    mask_i = idx_i < INTERMEDIATE_SIZE

    idx_h = tl.arange(0, HIDDEN_SIZE)
    mask_h = idx_h < HIDDEN_SIZE

    a = tl.load(
        a_ptr
        + idx_b[:, None, None] * stride_ab
        + idx_n[None, :, None] * stride_an
        + idx_h[None, None, :] * stride_ah,
        mask=mask_b[:, None, None] & mask_n[None, :, None] & mask_h[None, None, :],
    )

    b = tl.load(
        b_ptr
        + idx_b[:, None, None] * stride_bb
        + idx_h[None, :, None] * stride_bh
        + idx_i[None, None, :] * stride_bi,
        mask=mask_b[:, None, None] & mask_h[None, :, None] & mask_i[None, None, :],
    )

    c = tl.dot(a, b)

    tl.store(
        c_ptr
        + idx_b[:, None, None] * stride_cb
        + idx_n[None, :, None] * stride_cn
        + idx_i[None, None, :] * stride_ci,
        c,
        mask=mask_b[:, None, None] & mask_n[None, :, None] & mask_i[None, None, :],
    )


def init_tensors(
    batch_size: int, n: int, hidden_size: int, intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intype = torch.float16 if HALF_FLOAT else torch.float32
    a = torch.rand((batch_size, n, hidden_size), dtype=intype)
    b = torch.rand((batch_size, hidden_size, intermediate_size), dtype=intype)
    c = torch.empty((batch_size, n, intermediate_size), dtype=torch.float32)
    return a, b, c


def golden(
    batch_size: int, n: int, hidden_size: int, intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c = init_tensors(batch_size, n, hidden_size, intermediate_size)
    c = a.to(torch.float32) @ b.to(torch.float32)
    return a, b, c


def hex_baseline(a: torch.Tensor, b: torch.Tensor, enableHexKL: bool = False):
    batch_size = a.shape[0]
    n = a.shape[1]
    intermediate_size = b.shape[2]
    hidden_size = a.shape[2]
    assert a.shape[2] == b.shape[1]
    assert a.dtype == b.dtype

    c = torch.empty((batch_size, n, intermediate_size), dtype=torch.float32)

    assert n % 2 == 0
    assert intermediate_size % 2 == 0

    bsn = n
    bsi = intermediate_size
    ncpu = 1

    matmul_kernel[(ncpu,)](
        a,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b,
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c,
        c.stride(0),
        c.stride(1),
        c.stride(2),
        N=n,
        BATCH_SIZE=batch_size,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_SIZE_N=bsn,
        BLOCK_SIZE_I=bsi,
        enableMultiThreading=not enableHexKL,
        enableVTCMTiling=False,
        enableDoubleBuffering=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableHexKL=enableHexKL,
    )

    return c


def test_hexkl():
    a, b, gold = golden(BATCH_SIZE, N, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    baseline = hex_baseline(a, b)
    assert torch.allclose(gold, baseline)
    hmx = hex_baseline(a, b, enableHexKL=True)
    assert torch.allclose(gold, hmx, atol=1e-2)


if __name__ == "__main__":
    test_hexkl()
