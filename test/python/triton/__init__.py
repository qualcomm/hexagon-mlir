# ===- __init__.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
from typing import Callable, Optional

import pytest
import torch

# ===------------------------------------------------------------------------===
# GLOBALS
# ===------------------------------------------------------------------------===

KB = 1024
MB = 1024**2
GB = 1024**3

T = int(os.environ.get("NUM_THREADS", "1"))

C = int(os.environ.get("NUM_CORES", "1"))

DTYPE_STR = {
    torch.float16: "f16",
    torch.float32: "f32",
    torch.bool: "i1",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
}

DEFAULT_ATOL = {
    torch.float32: 1e-5,
    torch.float16: 1e-2,
}

# ===------------------------------------------------------------------------===
# FUNCTIONS
# ===------------------------------------------------------------------------===


def dtype_to_str(dtype) -> str:
    """Given a PyTorch dtype, return its string representation."""
    return DTYPE_STR.get(dtype, str(dtype))


def human_size(size: int) -> str:
    """Given a shape dimension return it as a string with G, M, or K suffix."""
    if (size % GB) == 0:
        return f"{size // GB}G"
    if (size % MB) == 0:
        return f"{size // MB}M"
    if (size % KB) == 0:
        return f"{size // KB}K"
    return str(size)


def shape_to_str(shape: tuple[int, ...]) -> str:
    """Given a shape represented as a tuple return its string representation."""
    return "x".join(human_size(dim) for dim in shape)


def dim_to_str(dim: int) -> str:
    """Given a dimension index, return its string represntation."""
    if dim == -1:
        return "lastdim"
    return f"dim{dim}"


def allclose(a, b, atol: float | dict | None = None) -> bool:
    """Wrapper around `torch.allclose` that uses default atol when not specified."""
    assert a.dtype == b.dtype, "Both tensors must have same dtype"

    if isinstance(atol, dict):
        atol = atol.get(a.dtype)

    if atol is None:
        atol = DEFAULT_ATOL.get(a.dtype)

    return torch.allclose(a, b, atol=atol)


# ===------------------------------------------------------------------------===
# DECORATORS
# ===------------------------------------------------------------------------===


def parametrize(*args, **kwargs):
    """Shorthand for pytest.mark.parametrize."""
    return pytest.mark.parametrize(*args, **kwargs)


def skip(*args, **kwargs):
    """Shorthand for pytest.mark.skip."""
    return pytest.mark.skip(*args, **kwargs)


def skipif(*args, **kwargs):
    """Shorthand for pytest.mark.skipif."""
    return pytest.mark.skipif(*args, **kwargs)


def dsp(fn: Optional[Callable] = None):
    """Decorator that adds the `dsp` marker to a function."""
    mark = pytest.mark.dsp
    # Called as `@dsp` (no parens)
    if fn is not None:
        return mark(fn)
    # Called as `@dsp()`
    return mark


def fixme(fn: Optional[Callable] = None):
    """Decorator that unconditionally skips a test due to CI failures."""
    mark = pytest.mark.skip(reason="FIXME: Test failing on CI.")
    # Called as `@fixme` (no parens)
    if fn is not None:
        return mark(fn)
    # Called as `@fixme()`
    return mark


def parametrize_bool(name: str):
    """Decorator that parametrizes a test with a boolean flag."""
    return parametrize(name, [False, True], ids=[f"{name}off", f"{name}on"])


def parametrize_dtypes(*dtypes, xfail: dict | None = None):
    """Decorator that parametrizes a test with Torch dtypes.

    `xfail` maps a dtype to an xfail reason, for dtypes whose numeric check is
    known to fail. Those params are marked non-strict, since an accuracy
    failure can squeak under tolerance on lucky random data.
    """
    xfail = xfail or {}
    params = []
    for dtype in dtypes:
        name = dtype_to_str(dtype)
        marks = [getattr(pytest.mark, name)]
        if reason := xfail.get(dtype):
            marks.append(pytest.mark.xfail(reason=reason, strict=False))
        params.append(pytest.param(dtype, marks=marks, id=name))
    return parametrize("dtype", params)


def parametrize_shapes(*shapes: tuple[int, ...]):
    """Decorator that parametrizes a test with given shapes."""
    for i, shape in enumerate(shapes):
        if not isinstance(shape, tuple):
            raise ValueError(f"Shape[{i}] '{shape}' is not a tuple.")
    return parametrize("shape", list(shapes), ids=shape_to_str)


def parametrize_dims(*dims: int):
    """Decorator that parametrizes a test with given dimension indices."""
    return parametrize("dim", list(dims), ids=dim_to_str)


def parametrize_reduces(*reduces: str):
    """Decorator that parametrizes a test with given reduction methods."""
    return parametrize("reduce", list(reduces), ids=lambda x: x)
