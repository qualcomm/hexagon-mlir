# ===- libdevice.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import math as _math

from triton.backends.qcom_hexagon_backend import knobs
from triton.language import core

import triton

# _USE_TANH_APPROX_DEFAULT is the compile-time default for the use_tanh_approx
# constexpr on erf()/gelu() below (see parse_options in compiler.py for what the
# HEXAGON_ENABLE_ACCURACY env means). It is resolved once at module import time
# because a @triton.jit body cannot read the environment at trace time -- the
# value has to be baked in as a Python constant when the module is imported.
#
# Reading the env here is what lets the flag reach inductor-generated kernels:
# inductor emits calls like libdevice.erf(x) / libdevice.gelu(x) with no
# use_tanh_approx argument, so they pick up this module-level default. Hand-
# written kernels (and the tests, which exercise both paths in one process) can
# still override it per call by passing use_tanh_approx as an explicit constexpr
# kwarg.
_ENABLE_ACCURACY = knobs.hexagon.enable_accuracy
# constexpr default for use_tanh_approx: True == fast tanh path.
_USE_TANH_APPROX_DEFAULT = not _ENABLE_ACCURACY


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_exp2_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_exp2_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_log2_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_log2_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


def _pow_const(value):
    # value as a float when it is a compile-time constant, else None.
    value = core._unwrap_if_constexpr(value)
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


@core.extern
def _pow_hvx(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0, arg1],
        {
            (core.dtype("fp32"), core.dtype("fp32")): (
                "qhmath_hvx_pow_af",
                core.dtype("fp32"),
            ),
            (core.dtype("fp16"), core.dtype("fp16")): (
                "qhmath_hvx_pow_ahf",
                core.dtype("fp16"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )


def _pow_handle_edge_cases(raw, base, exponent, arg0, _semantic):
    """Patch IEEE-754 special values into a raw qhmath_hvx_pow result.

    WORKAROUND: qhmath_hvx_pow_{af,ahf} computes exp2(y * log2(|x|)) and
    reapplies the base's sign only for odd integer exponents. It has no
    special-value handling, so two regions disagree with torch.pow:

      1. base < 0, non-integer exponent -> NaN, not |base|**exp
         (pow(-2.0, 2.5) returns 5.65625).
      2. base == 0, exponent < 0 -> +/-inf, not garbage
         (pow(0.0, -2.0) returns -0.25).

    A zero base also needs the sign restored for odd exponents, which the
    routine drops (pow(-0.0, 1.0) returns +0.0, want -0.0).

    ``arg0`` is the caller's unconverted base, kept so a compile-time constant
    is still visible after ``to_tensor`` has run.

    This function and its call site can be removed once the QHL routine handles
    special values.
    """

    def splat(value):
        return core.full(raw.shape, value, raw.dtype, _semantic=_semantic)

    const_base = _pow_const(arg0)
    if const_base is not None and const_base > 0.0:
        return raw

    zero = splat(0.0)
    is_int = _semantic.equal(floor(exponent, _semantic=_semantic), exponent)
    half = _semantic.mul(exponent, splat(0.5), True)
    is_odd = _semantic.and_(
        is_int,
        _semantic.not_(_semantic.equal(floor(half, _semantic=_semantic), half)),
    )

    finite = _semantic.where(
        _semantic.and_(_semantic.less_than(base, zero), _semantic.not_(is_int)),
        splat(float("nan")),
        raw,
    )

    magnitude = _semantic.where(
        _semantic.less_than(exponent, zero), splat(float("inf")), raw
    )
    if const_base is not None:
        negative_base = _math.copysign(1.0, const_base) < 0.0
        signed = _semantic.minus(magnitude) if negative_base else magnitude
    else:
        int_dtype = core.dtype(f"int{base.dtype.primitive_bitwidth}")
        sign_bit = _semantic.less_than(
            _semantic.bitcast(base, int_dtype),
            core.full(base.shape, 0, int_dtype, _semantic=_semantic),
        )
        signed = _semantic.where(sign_bit, _semantic.minus(magnitude), magnitude)

    return _semantic.where(
        _semantic.equal(base, zero),  # true for -0.0 too
        _semantic.where(is_odd, signed, magnitude),
        finite,
    )


@core.builtin
def pow(arg0, arg1, _semantic=None):
    base = _semantic.to_tensor(arg0)
    exponent = _semantic.to_tensor(arg1)
    raw = _pow_hvx(base, exponent, _semantic=_semantic)
    return _pow_handle_edge_cases(raw, base, exponent, arg0, _semantic=_semantic)


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


@triton.jit
def _erf_tanh_approx(x):
    # Fast path: approximate erf with the existing tanh extern:
    #     erf(x) ~= tanh(a * x + b * x**3)
    # with a = 2/sqrt(pi) (matching erf'(0)) and b fitted so the composite also
    # reproduces the tanh-GELU formulation 0.5*x*(1+erf(x/sqrt2)). Max abs error
    # vs the true erf is < 4e-4 over [-4, 4], well inside test tolerances.
    # x is the fp32-promoted input. The NaN guard is applied by the erf() caller
    # (both paths need it), so this returns the raw composite.
    a = 1.1283791670955126
    b = 0.10091094891335171
    return tanh(a * x + b * x * x * x)


@triton.jit
def _erf_accurate(x):
    # Accurate path: a direct port of the eager-mode erf kernel, so the triton
    # erf is numerically consistent with torch.erf on device. Two Horner
    # polynomials selected on
    # |x|; coefficients are copied verbatim from the eager kernel. x is the
    # fp32-promoted input. Measured max abs error vs true erf is ~8e-4 (the
    # polynomial peaks near |x|=2 and the 3.925 saturation clamp, plus on-device
    # qf32 rounding), tighter and more eager-consistent than the tanh composite,
    # at the cost of more HVX cycles. NaN is not handled here (the vcmp/vmux
    # selects flush it); the erf() caller applies the shared NaN guard.
    x_abs = core.where(x < 0.0, -x, x)
    r_large_input = core.minimum(x_abs, 3.925)
    # Horner (highest-degree coefficient first): r_large_coefficients evaluated
    # by horner_evaluate.
    r_large = -0.000811280500071568
    r_large = r_large * r_large_input + 0.01801244220953157
    r_large = r_large * r_large_input + -0.1609129525628144
    r_large = r_large * r_large_input + 0.7462932394233366
    r_large = r_large * r_large_input + -1.9059661498611058
    r_large = r_large * r_large_input + 2.5529950667638293
    r_large = r_large * r_large_input + -0.40684631957764783
    # copysign(r_large, x): erf is odd, so restore the input's sign.
    r_large = core.where(x < 0.0, -r_large, r_large)
    # r_small_coefficients, evaluated on x_abs.
    r_small = 0.019806570111091056
    r_small = r_small * x_abs + 0.12073226818702078
    r_small = r_small * x_abs + -0.43963558651654566
    r_small = r_small * x_abs + 0.015055581842623345
    r_small = r_small * x_abs + 1.1269725333344989
    r_small = r_small * x_abs + 3.108369669491129e-05
    r_small = core.where(x < 0.0, -r_small, r_small)
    return core.where(x_abs > 0.921875, r_large, r_small)


@triton.jit
def erf(
    arg0,
    propagate_nan: core.constexpr = True,
    use_tanh_approx: core.constexpr = _USE_TANH_APPROX_DEFAULT,
):
    # Hexagon has no native erf HVX symbol (no qhmath_hvx_erf_af/_ahf). Two
    # implementations are provided (see module header for the env/constexpr
    # switch via HEXAGON_ENABLE_ACCURACY):
    #   use_tanh_approx=True  (default, perf): tanh(a*x + b*x**3) composite.
    #   use_tanh_approx=False (accurate):      the eager erf polynomial.
    #
    # Both promote to fp32 first: the tanh symbols are fp32/fp16 only and the
    # polynomial coefficients need full fp32 precision, and bf16/fp16 would lose
    # mantissa bits if evaluated natively. Compute in fp32, then demote back to
    # the input dtype.
    #
    # NaN guard (propagate_nan): neither path propagates a NaN input on its own
    # -- the QHL tanh returns +/-1 for NaN, and the accurate path's vcmp/vmux
    # selects flush it -- so erf(nan) would come back ~1.0 rather than nan,
    # diverging from eager erf and CPU erf. Apply one guard here for both
    # paths. Key the mask on the ORIGINAL input (arg0), not the promoted x: for
    # fp16 the promote+fuse path can drop the NaN before the compare, so test
    # arg0 directly and select the promoted NaN through. Keying on arg0 (rather
    # than an intermediate) means erf(+/-inf) still saturates to +/-1: the
    # transient NaN the math produces for an inf input is not a real input NaN.
    # isnan is the upstream @core.builtin (x != x); it lowers to arith.cmpf,
    # which HexagonAddFastMathPass deliberately skips (to preserve isNaN checks),
    # and compiles to a vectorized unsigned-word compare (q = vcmp.gt(.uw))
    # feeding a vmux -- the same HVX predicate a hand-written intrinsic would
    # emit, not a scalar fallback.
    x = arg0.to(core.float32)
    if use_tanh_approx:
        r = _erf_tanh_approx(x)
    else:
        r = _erf_accurate(x)
    if propagate_nan:
        r = core.where(isnan(arg0), x, r)
    return r.to(arg0.dtype)


@triton.jit
def gelu(arg0, use_tanh_approx: core.constexpr = _USE_TANH_APPROX_DEFAULT):
    # GELU with the same tanh(perf)/accurate switch as erf (see module header
    # for HEXAGON_ENABLE_ACCURACY env/constexpr switch).
    #   use_tanh_approx=True  (default, perf): the tanh-GELU approximation
    #       0.5*x*(1 + tanh(0.797885*x + 0.035677*x**3)).
    #   use_tanh_approx=False (accurate):      the exact form
    #       0.5*x*(1 + erf(x/sqrt2)), routed through the accurate erf so it
    #       matches torch.nn.functional.gelu(approximate="none").
    # Compute in fp32 and demote back, matching erf's dtype handling.
    # NaN guard: Both paths require explicit NaN propagation. On the tanh path
    # the underlying HVX tanh extern does not propagate NaN. On the accurate
    # path the erf polynomial's vcmp/vmux selects flush NaN. Apply one guard
    # here for both paths to ensure gelu(nan)=nan, matching CPU behavior.
    # Note: gelu(x) = 0.5*x*(1+f(x)) is an inf*0 indeterminate form at x=-inf
    # (f(x) saturates to -1), and CPU F.gelu itself returns nan at the
    # infinities (both approximate modes) -- so no attempt is made to special-
    # case them here; only the well-defined nan-input case is guarded.
    x = arg0.to(core.float32)
    if use_tanh_approx:
        inner = tanh(0.797884560802865 * x + 0.0356774081114104 * x * x * x)
        r = 0.5 * x * (1.0 + inner)
    else:
        # 1/sqrt(2) = 0.7071067811865476
        e = _erf_accurate(x * 0.7071067811865476)
        r = 0.5 * x * (1.0 + e)
    # Propagate NaN: key on the original input (arg0), not the promoted x.
    r = core.where(isnan(arg0), x, r)
    return r.to(arg0.dtype)


def _copysign_const(value):
    # sign of value as +1.0 / -1.0 when it is a compile-time constant scalar,
    # else None. Mirrors _pow_const: lets a constant sign source fold at trace
    # time instead of emitting a runtime sign-bit extraction.
    value = core._unwrap_if_constexpr(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _math.copysign(1.0, float(value))
    return None


@core.builtin
def _copysign(mag, sign_src, _semantic=None):
    # copysign(mag, sign_src): |mag| carrying sign_src's sign. Taken via bit ops
    # (|mag| OR sign_src's sign bit) rather than `mag * where(sign_src < 0, -1,
    # 1)`: on this backend a float sign reconstructed with a compare/select
    # after a @core.extern call (exp/log2/sqrt) is dropped, whereas the bitwise
    # form lowers correctly.
    #
    # When sign_src is a compile-time constant its sign is resolved in Python
    # (as pow() does for a constant base via _pow_const), so no runtime sign-bit
    # extraction is emitted and only |mag| is computed on device.
    #
    # The bit masks are keyed on mag's native width (the sole caller, erfinv,
    # promotes to fp32 first, but the masks are chosen dynamically so a 16-bit
    # mag stays correct).
    mag = _semantic.to_tensor(mag)
    if mag.dtype.primitive_bitwidth == 16:
        int_dtype, float_dtype = core.uint16, core.float16
        abs_mask, sign_mask = 0x7FFF, 0x8000
    else:
        int_dtype, float_dtype = core.uint32, core.float32
        abs_mask, sign_mask = 0x7FFFFFFF, 0x80000000
    abs_bits = _semantic.and_(
        _semantic.bitcast(mag, int_dtype),
        core.full(mag.shape, abs_mask, int_dtype, _semantic=_semantic),
    )
    abs_mag = _semantic.bitcast(abs_bits, float_dtype)
    const_sign = _copysign_const(sign_src)
    if const_sign is not None:
        return _semantic.minus(abs_mag) if const_sign < 0.0 else abs_mag
    sign_src = _semantic.to_tensor(sign_src)
    sign_bit = _semantic.and_(
        _semantic.bitcast(sign_src, int_dtype),
        core.full(sign_src.shape, sign_mask, int_dtype, _semantic=_semantic),
    )
    return _semantic.bitcast(_semantic.or_(abs_bits, sign_bit), float_dtype)


@triton.jit
def _erfinv_core(y):
    # Pavlis (1996) piecewise rational approximation + one guarded Newton step.
    # y is the fp32-promoted input; both branches are evaluated and selected with
    # where (no divergent control flow). erfinv is odd: the magnitude is
    # computed on |y| and the input's sign is copied back with _copysign at the
    # end. Boundary/out-of-domain handling is the erfinv() caller's job.
    ay = core.where(y < 0.0, -y, y)
    z = ay * ay
    # Inner region |y| <= 0.7: |y| * P(z) / Q(z).
    p = -0.140543331
    p = p * z + 0.914624893
    p = p * z + -1.645349621
    p = p * z + 0.886226899
    q = 0.012229801
    q = q * z + -0.329097515
    q = q * z + 1.442710462
    q = q * z + -2.118377725
    q = q * z + 1.0
    m_inner = ay * (p / q)
    # Outer region |y| > 0.7: P(w) / Q(w), w = sqrt(-ln((1-|y|)/2)).
    ln2 = 0.6931471805599453
    neg_log = -(log2((1.0 - ay) * 0.5) * ln2)
    w = sqrt(neg_log)
    pc = 1.641345311
    pc = pc * w + 3.429567803
    pc = pc * w + -1.624906493
    pc = pc * w + -1.970840454
    qd = 1.637067800
    qd = qd * w + 3.543889200
    qd = qd * w + 1.0
    m_outer = pc / qd
    m = core.where(ay < 0.7, m_inner, m_outer)
    # Newton refinement m -= (erf(m) - |y|) / (2/sqrt(pi) * exp(-m^2)), guarded
    # to m < 1 where erf'(m) is not tiny.
    correction = (_erf_accurate(m) - ay) / (1.1283791670955126 * exp(-(m * m)))
    m = m - core.where(m < 1.0, correction, 0.0)
    # Reapply the input's sign (odd function).
    return _copysign(m, y)


@triton.jit
def erfinv(arg0):
    # Hexagon has no native erfinv HVX symbol; implement the erfinv
    # algorithm (see _erfinv_core). Promote to fp32 (the rational coefficients
    # need full precision), then demote back to the input dtype.
    x = arg0.to(core.float32)
    r = _erfinv_core(x).to(arg0.dtype)
    # Domain (-1, 1): |y| == 1 -> +/-inf, |y| > 1 -> NaN, NaN input -> NaN.
    # Match torch.erfinv. |arg0| and the signed inf are built with bit ops (mask
    # off / set the sign bit) in the INPUT's native width, not a float
    # compare/select on the fp32-promoted value:
    #   - a float `x < 0` compare after the extern-heavy _erfinv_core is
    #     miscompiled on this backend, so abs is taken bitwise instead;
    #   - bitcasting fp16-origin data through the fp32/uint32 path also
    #     miscompiles (any +/-inf built that way comes back NaN), so the fixup
    #     runs on arg0 in its native bit width (uint16 for fp16, uint32 for fp32).
    bitwidth: core.constexpr = arg0.dtype.primitive_bitwidth
    if bitwidth == 16:
        ai = arg0.to(core.uint16, bitcast=True)
        abs_a = (ai & 0x7FFF).to(core.float16, bitcast=True)
        signed_inf = ((ai & 0x8000) | 0x7C00).to(core.float16, bitcast=True)
    else:
        ai = arg0.to(core.uint32, bitcast=True)
        abs_a = (ai & 0x7FFFFFFF).to(core.float32, bitcast=True)
        signed_inf = ((ai & 0x80000000) | 0x7F800000).to(core.float32, bitcast=True)
    r = core.where(abs_a == 1.0, signed_inf.to(arg0.dtype), r)
    r = core.where(abs_a > 1.0, float("nan"), r)
    r = core.where(isnan(arg0), arg0, r)
    return r


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
            (core.dtype("fp32"),): ("qhmath_hvx_exp_af_ieee", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_exp_ahf_ieee", core.dtype("fp16")),
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


@core.extern
def sigmoid(arg0, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("qhmath_hvx_sigmoid_af", core.dtype("fp32")),
            (core.dtype("fp16"),): ("qhmath_hvx_sigmoid_ahf", core.dtype("fp16")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.builtin
def isnan(arg0, _semantic=None):
    # A value is NaN iff it does not equal itself.
    return _semantic.not_equal(arg0, arg0)
