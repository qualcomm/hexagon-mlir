# ===- test_vec_add.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# This test is created separately from test_vec_add.py to keep test_vec_add.py
# as simple as possible even though more code reuse would be possible by
# parametrizing the usage of the aligned allocator
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

from . import fixme

triton.runtime.driver.set_active(HexagonDriver())
BLOCK_SIZE = 131072 * 4


# redefining kernel as importing the kernel from test_vec_add.py
# would create a dependency on a different test module
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def aligned_torch_tensor(size, dtype=torch.float16, alignment=256):
    """
    Allocate a PyTorch tensor with 128-byte alignment by directly operating on data_ptr.

    This function creates a tensor that is guaranteed to be aligned to the specified
    alignment boundary without requiring device=cuda specification.

    Args:
        size: Number of elements in the tensor
        dtype: PyTorch data type (default: torch.float16)
        alignment: Alignment boundary in bytes (default: 128)

    Returns:
        A PyTorch tensor with data_ptr aligned to the specified boundary
    """
    # Calculate element size in bytes
    element_size = torch.tensor([], dtype=dtype).element_size()

    # Allocate extra elements to ensure we can find an aligned region
    # We need (alignment / element_size) extra elements to guarantee alignment
    extra_elements = alignment // element_size
    total_size = size + extra_elements

    # Allocate the oversized tensor
    buf = torch.empty(total_size, dtype=dtype)

    # Get the data pointer and calculate offset needed for alignment
    data_ptr = buf.data_ptr()
    offset_bytes = data_ptr % alignment

    if offset_bytes == 0:
        # Already aligned, just slice to the requested size
        aligned_tensor = buf[:size]
    else:
        # Calculate how many elements to skip to reach alignment
        padding_bytes = alignment - offset_bytes
        padding_elements = padding_bytes // element_size

        # Return a slice starting at the aligned position
        aligned_tensor = buf[padding_elements : padding_elements + size]

    return aligned_tensor


@fixme
def test_vec_add():
    # Using aligned_torch_tensor (direct PyTorch approach)
    x = aligned_torch_tensor(BLOCK_SIZE, dtype=torch.float32, alignment=128)
    y = aligned_torch_tensor(BLOCK_SIZE, dtype=torch.float32, alignment=128)
    output = aligned_torch_tensor(BLOCK_SIZE, dtype=torch.float32, alignment=128)

    print("START: ptrs are aligned to 128")
    print(f"x is aligned: {x.data_ptr() % 128 == 0}")
    print(f"y is aligned: {y.data_ptr() % 128 == 0}")
    print(f"output is aligned: {output.data_ptr() % 128 == 0}")
    print("END: ptrs are aligned to 128")

    # best performance currently is with just MT on.
    # Rest are enabled for correctness checks.
    add_kernel[(1,)](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
        enableDoubleBuffering=False,
        specializeKernelToArgAlignment=True,
    )
    reference = x + y
    assert torch.allclose(output, reference)
