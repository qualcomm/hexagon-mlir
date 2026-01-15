# ===- libdevice.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from triton.language import core


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_rsqrt_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_rsqrt_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_tanh_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_tanh_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_cos_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_cos_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_sin_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_sin_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_tan_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_tan_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_acos_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_acos_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_asin_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_asin_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_atan_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_atan_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_ceil_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_ceil_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_floor_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_floor_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_exp_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_exp_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_sqrt_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_sqrt_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
