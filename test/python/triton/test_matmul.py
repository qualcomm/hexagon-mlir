# ===- test_matmul.py -------------------------------------------------------===
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

triton.runtime.driver.set_active(HexagonDriver())

# 2e-1 is the minimum ATOL where the matmul test succeeds
ATOL = 2e-1


@pytest.mark.parametrize("num_rows", [1024])
@pytest.mark.parametrize("num_cols", [512])
@pytest.mark.parametrize("num_inner", [64])
def test_matmul(num_rows, num_cols, num_inner):
    print(
        f"Running with num-rows: {num_rows}, "
        f"Running with num-cols: {num_cols}, "
        f"Running with num-inner: {num_inner}, "
    )
    print("-" * 100)

    @triton.jit
    def matmul(
        A,
        B,
        C,
        N_ROWS: tl.constexpr,
        N_COLUMNS: tl.constexpr,
        N_INNER: tl.constexpr,
    ):
        A_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(N_ROWS, N_INNER),
            strides=(N_INNER, 1),
            offsets=(0, 0),
            block_shape=(N_ROWS, N_INNER),
            order=(1, 0),
        )
        B_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(N_INNER, N_COLUMNS),
            strides=(N_COLUMNS, 1),
            offsets=(0, 0),
            block_shape=(N_INNER, N_COLUMNS),
            order=(1, 0),
        )
        C_block_ptr = tl.make_block_ptr(
            base=C,
            shape=(N_ROWS, N_COLUMNS),
            strides=(N_COLUMNS, 1),
            offsets=(0, 0),
            block_shape=(N_ROWS, N_COLUMNS),
            order=(1, 0),
        )
        q = tl.load(A_block_ptr)
        k_t = tl.load(B_block_ptr)
        qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
        tl.store(C_block_ptr, qk)

    mat_A = torch.rand((num_rows, num_inner), dtype=torch.float16)
    mat_B = torch.rand((num_inner, num_cols), dtype=torch.float16)
    mat_C = torch.zeros((num_rows, num_cols), dtype=torch.float16)

    matmul[(1,)](
        mat_A,
        mat_B,
        mat_C,
        num_rows,
        num_cols,
        num_inner,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )

    # Perform the matrix multiplication
    reference = torch.matmul(mat_A, mat_B)
    assert torch.allclose(mat_C, reference, atol=ATOL)
