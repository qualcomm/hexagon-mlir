# This test is implemented with a block pointer construct to simulate a 4D tensor operation
# and was motivated by the need to test standalone HVX Croutonization, which is triggered
# on matching FP16 4D tensor patterns. The pass can be inspected in ForceHVXCrouton.cpp

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())


N, H, W, C = 16, 16, 128, 2


@triton.jit
def tanh(x):
    result = tl.exp((-2 * x).to(tl.float32)).to(x.dtype)
    result = (1 - result) / (1 + result)
    return result


@triton.jit
def gelu_bptr_kernel(
    x_ptr,
    output_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
):

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, H, W, C),
        strides=(H * W * C, W * C, C, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(N, H, W, C),
        order=(3, 2, 1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, H, W, C),
        strides=(H * W * C, W * C, C, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(N, H, W, C),
        order=(3, 2, 1, 0),
    )

    x = tl.load(x_block_ptr)
    output = 0.5 * x * (1 + tanh(0.797885 * x + 0.035677 * x * x * x)).to(tl.float16)
    tl.store(output_block_ptr, output)


def test_gelu_bptr():
    x = torch.rand(N, H, W, C, dtype=torch.float16)
    output = torch.empty_like(x, dtype=torch.float16)

    gelu_bptr_kernel[(1,)](
        x,
        output,
        N,
        H,
        W,
        C,
    )
