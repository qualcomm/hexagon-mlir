//===- IntrinsicsF8E4M3Fstar.c   - HVX FP8/FP16 conversion routines
//------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Contains HVX helper routines to convert between FP8 (E4M3FN/E4M3FNUZ) and
// FP16 encodings, including special-case handling for FN/FNUZ semantics.
//
//===----------------------------------------------------------------------===//

#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__HVX_ARCH__) && (__HVX_ARCH__ >= 81)

__attribute__((used, always_inline)) HVX_VectorPair
convert_fp8_fp16_fnuz(HVX_Vector in) {
  HVX_Vector mod_in = Q6_Vb_vshuff_Vb(in);
  HVX_VectorPair out = Q6_Wqf16_equals_V(mod_in);
  HVX_Vector out_lo = Q6_V_lo_W(out);
  HVX_Vector out_hi = Q6_V_hi_W(out);
  // Correct E4M3FN decode bias to E4M3FNUZ semantics (scale by 0.5).
  HVX_Vector half_scale = Q6_Vh_vsplat_R(0x3800);
  out_lo = Q6_Vqf16_vmpy_Vqf16Vhf(out_lo, half_scale);
  out_hi = Q6_Vqf16_vmpy_Vqf16Vhf(out_hi, half_scale);
  HVX_Vector out_lo_hf = Q6_Vhf_equals_Vqf16(out_lo);
  HVX_Vector out_hi_hf = Q6_Vhf_equals_Vqf16(out_hi);
  return Q6_W_vcombine_VV(out_hi_hf, out_lo_hf);
}

__attribute__((used, always_inline)) HVX_Vector
convert_fp16_fp8_fnuz(HVX_VectorPair in_vhf_pair) {
  HVX_Vector in_lo_vhf = Q6_V_lo_W(in_vhf_pair);
  HVX_Vector in_hi_vhf = Q6_V_hi_W(in_vhf_pair);
  HVX_Vector qf16_lo = Q6_Vqf16_equals_Vhf(in_lo_vhf);
  HVX_Vector qf16_hi = Q6_Vqf16_equals_Vhf(in_hi_vhf);
  // Reciprocal correction for encode path: E4M3FNUZ -> E4M3FN bridge (scale
  // by 2.0).
  HVX_Vector two_scale = Q6_Vh_vsplat_R(0x4000);
  qf16_lo = Q6_Vqf16_vmpy_Vqf16Vhf(qf16_lo, two_scale);
  qf16_hi = Q6_Vqf16_vmpy_Vqf16Vhf(qf16_hi, two_scale);
  HVX_Vector f8_lo = Q6_V_equals_Vqf16(qf16_lo);
  HVX_Vector f8_hi = Q6_V_equals_Vqf16(qf16_hi);

  // Dense fp8 bytes: low-half lanes then high-half lanes
  return Q6_Vb_vpacke_VhVh(f8_hi, f8_lo);
}

__attribute__((used, always_inline)) HVX_VectorPair
convert_fp8_fp16_fn(HVX_Vector in) {
  HVX_Vector mod_in = Q6_Vb_vshuff_Vb(in);

  // Decode normally.
  HVX_VectorPair out = Q6_Wqf16_equals_V(mod_in);
  HVX_Vector out_lo_hf = Q6_Vhf_equals_Vqf16(Q6_V_lo_W(out));
  HVX_Vector out_hi_hf = Q6_Vhf_equals_Vqf16(Q6_V_hi_W(out));

  // Patch FN -0 lanes: fp8 byte 0x80 -> fp16 -0 (0x8000).
  union {
    HVX_Vector v;
    unsigned char b[128];
  } in_u;
  union {
    struct {
      HVX_Vector lo, hi;
    } vv;
    unsigned short h[128];
  } out_u;

  // IMPORTANT: use original input bytes for -0 detection.
  // Using mod_in can miss many 0x80 lanes after vshuff.
  in_u.v = in;
  out_u.vv.lo = out_lo_hf;
  out_u.vv.hi = out_hi_hf;

  for (int i = 0; i < 128; ++i) {
    if (in_u.b[i] == 0x80)
      out_u.h[i] = 0x8000;
  }

  return Q6_W_vcombine_VV(out_u.vv.hi, out_u.vv.lo);
}

__attribute__((used, always_inline)) HVX_Vector
convert_fp16_fp8_fn(HVX_VectorPair in_vhf_pair) {
  HVX_Vector in_lo_vhf = Q6_V_lo_W(in_vhf_pair);
  HVX_Vector in_hi_vhf = Q6_V_hi_W(in_vhf_pair);

  HVX_Vector qf16_lo = Q6_Vqf16_equals_Vhf(in_lo_vhf);
  HVX_Vector qf16_hi = Q6_Vqf16_equals_Vhf(in_hi_vhf);

  HVX_Vector f8_lo = Q6_V_equals_Vqf16(qf16_lo);
  HVX_Vector f8_hi = Q6_V_equals_Vqf16(qf16_hi);

  // Dense fp8 bytes: low-half lanes then high-half lanes.
  HVX_Vector packed = Q6_Vb_vpacke_VhVh(f8_hi, f8_lo);

  // Patch using original input bits (same pattern as convertfn_fp8_fp16).
  union {
    struct {
      HVX_Vector lo, hi;
    } vv;
    unsigned short h[128];
  } in_u;

  union {
    HVX_Vector v;
    unsigned char b[128];
  } out_u;

  in_u.vv.lo = in_lo_vhf;
  in_u.vv.hi = in_hi_vhf;
  out_u.v = packed;

  for (int i = 0; i < 128; ++i) {
    unsigned short h = in_u.h[i];

    // fp16 NaN: exp=0x1F and mantissa!=0  -> force canonical FN NaN byte.
    int is_nan = ((h & 0x7C00) == 0x7C00) && ((h & 0x03FF) != 0);
    if (is_nan) {
      out_u.b[i] = 0x7F;
      continue;
    }

    // fp16 -0 (0x8000) -> fp8 -0 (0x80).
    if (h == 0x8000) {
      out_u.b[i] = 0x80;
    }
  }

  return out_u.v;
}

// The FP8 conversion helpers below are gated because they rely on
// Q6_Wqf16_equals_V (Vdd32.qf16=Vu32.f8) a v81-only instruction
// that decodes an FP8 byte in hardware.

#else // !(defined(__HVX_ARCH__) && (__HVX_ARCH__ >= 81))

// E4M3FN/FNUZ share the S1E4M3 layout [S:EEEE:MMM], but differ in
// exponent bias and special-value encodings (NaN and -0 handling).
#define F8_MANT_BITS 3
#define F8_MANT_MAX 0x0007 // 3-bit mantissa mask; also fp16's dropped-bit count
#define F8_MANT_CARRY 0x0008 // mantissa carry threshold; min-normal code 0x08
#define F8_EXP_MAX 0x000F    // 4-bit exponent field max
#define F8_NAN_FN 0x007F     // E4M3FN sole NaN code (sign added)
#define F8_MAX_FINITE_FN 0x007E // E4M3FN max finite code (+/-448), saturation
#define F8_NAN_UZ 0x0080        // E4M3FNUZ sole NaN code

// ---- decode (fp8 -> fp16) ----
#define F8_BYTE_TO_HALF_SHIFT                                                  \
  8 // move the fp8 byte's sign bit to halfword bit 15
#define F8_EXP_MAN_MASK 0x7FFF // exp+mantissa after the << 8 (strips the sign)
#define F8_NAN_SHIFTED 0x7F00  // sign-stripped NaN byte 0x7F after the << 8
#define F8_MIN_NORMAL_CLZ 4    // clz of the smallest normal once shifted up
// After the << 8 the exponent LSB sits at bit 11; fp16's mantissa is 10 bits,
// so the fp8 fields drop into fp16's layout with a single right shift of 1.
#define F8_EXP_LSB_TO_HF_MANT ((F8_BYTE_TO_HALF_SHIFT + F8_MANT_BITS) - 10)
#define HF_NAN_FN 0x7F80 // fp16 NaN that E4M3FN NaN maps to (sign OR-ed back)
#define HF_NAN_UZ 0x7E00 // fp16 qNaN that E4M3FNUZ NaN maps to (sign dropped)
#define HF_SIGN_MASK 0x8000
#define HF_MANT_BITS 10
// Exponent rebias, fp8 -> fp16: FLOAT_EXPONENT_BIAS_HF(15) - fp8 bias.
#define HF_REBIAS_FN 8 // 15 - 7
#define HF_REBIAS_UZ 7 // 15 - 8

// ---- encode (fp16 -> fp8) ----
#define HF_EXP_MASK 0x001F
#define HF_MANT_MASK 0x03FF
#define HF_IMPLICIT_ONE 0x0400
#define HF_ROUND_HALF                                                          \
  0x0040 // halfway point of the 7 dropped fp16 mantissa bits
#define HF_SIGN_SHIFT 15
// Subnormal right-shift base: 16 - exp16 for FN (bias 7), 15 - exp16 for FNUZ.
#define HF_SUB_SHIFT_BASE_FN 0x0010
#define HF_SUB_SHIFT_BASE_UZ 0x000F

// NOTE: The algorithm is a plain-C port of the v68 helpers the eager-mode
// operator library already ships.

// Decode one half of the payload: 64 E4M3FNUZ bytes, already zero-extended
// into halfword lanes, into 64 fp16 lanes. Called once per half so the body is
// written once instead of being mirrored for _lo and _hi.
static inline __attribute__((always_inline)) HVX_Vector
decodeFp8HalfToF16Fnuz(HVX_Vector hbits) {
  // Splat constants. always_inline lets both halves share one splat each.
  HVX_Vector vZero = Q6_Vh_vsplat_R(0);
  HVX_Vector vByteShift = Q6_Vh_vsplat_R(F8_BYTE_TO_HALF_SHIFT);
  HVX_Vector vExpManMask = Q6_Vh_vsplat_R(F8_EXP_MAN_MASK);
  HVX_Vector vSignMask = Q6_Vh_vsplat_R(HF_SIGN_MASK);
  HVX_Vector vMinNormClz = Q6_Vh_vsplat_R(F8_MIN_NORMAL_CLZ);
  HVX_Vector vAlignShift = Q6_Vh_vsplat_R(F8_EXP_LSB_TO_HF_MANT);
  HVX_Vector vRebias = Q6_Vh_vsplat_R(HF_REBIAS_UZ);
  HVX_Vector vMantBits = Q6_Vh_vsplat_R(HF_MANT_BITS);
  HVX_Vector vNan = Q6_Vh_vsplat_R(HF_NAN_UZ);

  // Split the byte into sign and exponent+mantissa, aligned to halfword bit 15.
  HVX_Vector shifted = Q6_Vh_vasl_VhVh(hbits, vByteShift);
  HVX_Vector expMan = Q6_V_vand_VV(shifted, vExpManMask);
  HVX_Vector sign = Q6_V_vand_VV(shifted, vSignMask);

  // Renormalize subnormals: leading zeros beyond the min-normal count give the
  // left shift, and the same amount is subtracted from the rebiased exponent.
  HVX_Vector lz = Q6_Vuh_vcl0_Vuh(expMan);
  HVX_Vector mantShift = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(lz, vMinNormClz),
                                       Q6_Vh_vsub_VhVh(lz, vMinNormClz), vZero);

  // Drop the fp8 fields into fp16's layout and add the rebiased exponent.
  HVX_Vector aligned =
      Q6_Vh_vlsr_VhVh(Q6_Vh_vasl_VhVh(expMan, mantShift), vAlignShift);
  HVX_Vector expTerm =
      Q6_Vh_vasl_VhVh(Q6_Vh_vsub_VhVh(vRebias, mantShift), vMantBits);
  HVX_Vector nonZero = Q6_V_vor_VV(sign, Q6_Vh_vadd_VhVh(aligned, expTerm));

  // E4M3FNUZ has no -0: an all-zero magnitude is +0 whatever the sign bit,
  // which frees byte 0x80 (sign set, magnitude clear) to be the sole NaN code.
  HVX_Vector result =
      Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(expMan, vZero), vZero, nonZero);
  result = Q6_V_vmux_QVV(
      Q6_Q_vcmp_eq_VhVh(Q6_V_vor_VV(expMan, sign), vSignMask), vNan, result);

  return result;
}

// Decode 128 packed E4M3FNUZ bytes into 128 fp16 lanes (a vector pair).
// Bias 8, no negative zero, sole NaN code 0x80 -> fp16 qNaN 0x7E00 sign
// dropped.
__attribute__((used)) HVX_VectorPair convert_fp8_fp16_fnuz(HVX_Vector in) {
  // Widen the packed bytes into halfword lanes; vshuff pre-permutes so the pair
  // comes out in element order.
  HVX_VectorPair hbits_w = Q6_Wuh_vzxt_Vub(Q6_Vb_vshuff_Vb(in));

  return Q6_W_vcombine_VV(decodeFp8HalfToF16Fnuz(Q6_V_hi_W(hbits_w)),
                          decodeFp8HalfToF16Fnuz(Q6_V_lo_W(hbits_w)));
}

// Encode one half of the payload: 64 fp16 lanes into 64 E4M3FNUZ codes, each in
// the low byte of its halfword lane. Called once per half so the body is
// written once instead of being mirrored for _lo and _hi.
static inline __attribute__((always_inline)) HVX_Vector
encodeF16HalfToFp8Fnuz(HVX_Vector in) {
  // Splat constants. always_inline lets both halves share one splat each.
  // Several values serve double duty as shift amounts and comparison operands
  // (e.g. v7 is the mantissa right-shift, the sign left-shift, and the
  // "mantissa == 7" test).
  HVX_Vector v0 = Q6_Vh_vsplat_R(0);
  HVX_Vector v1 = Q6_Vh_vsplat_R(1);
  HVX_Vector v3 = Q6_Vh_vsplat_R(F8_MANT_BITS);
  HVX_Vector v7 = Q6_Vh_vsplat_R(F8_MANT_MAX);
  HVX_Vector v8 = Q6_Vh_vsplat_R(F8_MANT_CARRY);
  HVX_Vector v10 = Q6_Vh_vsplat_R(HF_MANT_BITS);
  HVX_Vector v15 = Q6_Vh_vsplat_R(F8_EXP_MAX);
  HVX_Vector vSignShift = Q6_Vh_vsplat_R(HF_SIGN_SHIFT);
  HVX_Vector vExpMask = Q6_Vh_vsplat_R(HF_EXP_MASK);
  HVX_Vector vMantMask = Q6_Vh_vsplat_R(HF_MANT_MASK);
  HVX_Vector vImplicitOne = Q6_Vh_vsplat_R(HF_IMPLICIT_ONE);
  HVX_Vector vRoundHalf = Q6_Vh_vsplat_R(HF_ROUND_HALF);
  HVX_Vector vRoundMask = Q6_Vh_vsplat_R(F8_NAN_FN); // low 7 dropped bits
  HVX_Vector vRebias = Q6_Vh_vsplat_R(HF_REBIAS_UZ);
  HVX_Vector vSubBase = Q6_Vh_vsplat_R(HF_SUB_SHIFT_BASE_UZ);
  HVX_Vector vNan = Q6_Vh_vsplat_R(F8_NAN_UZ);

  // Decompose the fp16 fields and rebias the exponent into fp8's range.
  HVX_Vector sign = Q6_Vh_vlsr_VhVh(in, vSignShift);
  HVX_Vector exp = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(in, v10), vExpMask);
  HVX_Vector mant = Q6_V_vand_VV(in, vMantMask);
  HVX_Vector signif = Q6_V_vor_VV(mant, vImplicitOne);
  HVX_Vector E8 = Q6_Vh_vsub_VhVh(exp, vRebias);

  // ---- Normal candidate (E8 >= 1): keep the top 3 mantissa bits, RNE the
  // low 7.
  HVX_Vector mantTop = Q6_Vh_vlsr_VhVh(mant, v7);
  HVX_Vector roundRem = Q6_V_vand_VV(mant, vRoundMask);
  HVX_VectorPred roundUp_q =
      Q6_Q_or_QQ(Q6_Q_vcmp_gt_VhVh(roundRem, vRoundHalf),
                 Q6_Q_and_QQ(Q6_Q_vcmp_eq_VhVh(roundRem, vRoundHalf),
                             Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(mantTop, v1), v1)));
  HVX_Vector mantTopR =
      Q6_Vh_vadd_VhVh(mantTop, Q6_V_vmux_QVV(roundUp_q, v1, v0));

  // A mantissa that rounds up out of 3 bits carries into the exponent.
  HVX_VectorPred carry_q = Q6_Q_vcmp_eq_VhVh(mantTopR, v8);
  mantTopR = Q6_V_vmux_QVV(carry_q, v0, mantTopR);
  HVX_Vector E8n = Q6_Vh_vadd_VhVh(E8, Q6_V_vmux_QVV(carry_q, v1, v0));
  HVX_Vector normalCode = Q6_V_vor_VV(Q6_Vh_vasl_VhVh(E8n, v3), mantTopR);

  // E4M3FNUZ's top code 0x7F is finite (240), so only a true exponent overflow
  // becomes NaN. A mantissa carry pushing E8n from 15 to 16 is caught here too.
  normalCode = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(E8n, v15), vNan, normalCode);

  // ---- Subnormal candidate (E8 <= 0): mant8 = signif >> (15 - exp16), RNE.
  HVX_Vector shift = Q6_Vh_vsub_VhVh(vSubBase, exp);
  HVX_Vector mant8 = Q6_Vh_vlsr_VhVh(signif, shift);
  HVX_Vector oneShifted = Q6_Vh_vasl_VhVh(v1, shift);
  HVX_Vector subRem = Q6_V_vand_VV(signif, Q6_Vh_vsub_VhVh(oneShifted, v1));
  HVX_Vector subHalf = Q6_Vh_vlsr_VhVh(oneShifted, v1);
  HVX_VectorPred subUp_q =
      Q6_Q_or_QQ(Q6_Q_vcmp_gt_VhVh(subRem, subHalf),
                 Q6_Q_and_QQ(Q6_Q_vcmp_eq_VhVh(subRem, subHalf),
                             Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(mant8, v1), v1)));
  HVX_Vector mant8R = Q6_Vh_vadd_VhVh(mant8, Q6_V_vmux_QVV(subUp_q, v1, v0));
  // Rounding up out of the subnormal range lands on the minimum normal (0x08).
  HVX_Vector subCode = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(mant8R, v7), v8, mant8R);

  // E8 < 1 selects the subnormal/underflow path.
  HVX_Vector code =
      Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(v1, E8), subCode, normalCode);

  // fp16 zero / subnormal (exp16 == 0) flushes to zero; Inf / NaN -> NaN code.
  code = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(exp, v0), v0, code);
  code = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(exp, vExpMask), vNan, code);

  // Apply the sign into bit 7, then force a zero result back to +0: E4M3FNUZ
  // cannot represent -0, so a negative underflow must not produce one. The NaN
  // code 0x80 is non-zero, so it keeps its fixed value.
  HVX_Vector withSign = Q6_V_vor_VV(code, Q6_Vh_vasl_VhVh(sign, v7));
  withSign = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(code, v0), v0, withSign);

  return withSign;
}

// Encode 128 fp16 lanes (a vector pair) into 128 packed E4M3FNUZ bytes.
// RNE rounding; bias 8 (max finite 240); overflow / Inf / NaN -> 0x80; fp16
// zero, subnormals and negative underflow -> +0 (the format has no -0).
__attribute__((used)) HVX_Vector
convert_fp16_fp8_fnuz(HVX_VectorPair in_vhf_pair) {
  // Gather the low byte of each halfword lane, preserving element order.
  return Q6_Vb_vpacke_VhVh(encodeF16HalfToFp8Fnuz(Q6_V_hi_W(in_vhf_pair)),
                           encodeF16HalfToFp8Fnuz(Q6_V_lo_W(in_vhf_pair)));
}

// Decode one half of the payload: 64 E4M3FN bytes, already zero-extended into
// halfword lanes, into 64 fp16 lanes. Called once per half so the body is
// written once instead of being mirrored for _lo and _hi.
static inline __attribute__((always_inline)) HVX_Vector
decodeFp8HalfToF16Fn(HVX_Vector hbits) {
  // Splat constants. always_inline lets both halves share one splat each.
  HVX_Vector vZero = Q6_Vh_vsplat_R(0);
  HVX_Vector vByteShift = Q6_Vh_vsplat_R(F8_BYTE_TO_HALF_SHIFT);
  HVX_Vector vExpManMask = Q6_Vh_vsplat_R(F8_EXP_MAN_MASK);
  HVX_Vector vSignMask = Q6_Vh_vsplat_R(HF_SIGN_MASK);
  HVX_Vector vMinNormClz = Q6_Vh_vsplat_R(F8_MIN_NORMAL_CLZ);
  HVX_Vector vAlignShift = Q6_Vh_vsplat_R(F8_EXP_LSB_TO_HF_MANT);
  HVX_Vector vRebias = Q6_Vh_vsplat_R(HF_REBIAS_FN);
  HVX_Vector vMantBits = Q6_Vh_vsplat_R(HF_MANT_BITS);
  HVX_Vector vNanShifted = Q6_Vh_vsplat_R(F8_NAN_SHIFTED);
  HVX_Vector vNan = Q6_Vh_vsplat_R(HF_NAN_FN);

  // Split the byte into sign and exponent+mantissa, aligned to halfword bit 15.
  HVX_Vector shifted = Q6_Vh_vasl_VhVh(hbits, vByteShift);
  HVX_Vector expMan = Q6_V_vand_VV(shifted, vExpManMask);
  HVX_Vector sign = Q6_V_vand_VV(shifted, vSignMask);

  // Renormalize subnormals: leading zeros beyond the min-normal count give the
  // left shift, and the same amount is subtracted from the rebiased exponent.
  HVX_Vector lz = Q6_Vuh_vcl0_Vuh(expMan);
  HVX_Vector mantShift = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(lz, vMinNormClz),
                                       Q6_Vh_vsub_VhVh(lz, vMinNormClz), vZero);

  // Drop the fp8 fields into fp16's layout and add the rebiased exponent.
  HVX_Vector aligned =
      Q6_Vh_vlsr_VhVh(Q6_Vh_vasl_VhVh(expMan, mantShift), vAlignShift);
  HVX_Vector expTerm =
      Q6_Vh_vasl_VhVh(Q6_Vh_vsub_VhVh(vRebias, mantShift), vMantBits);
  HVX_Vector nonZero = Q6_V_vor_VV(sign, Q6_Vh_vadd_VhVh(aligned, expTerm));

  // exp_man == 0 is +/-0 (E4M3FN keeps the sign, unlike FNUZ).
  HVX_Vector result =
      Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(expMan, vZero), sign, nonZero);

  // E4M3FN has no Inf: S.1111.111 is the sole NaN. After the << 8 the
  // sign-stripped NaN byte 0x7F lands as exp_man == 0x7F00; OR the sign back in
  // so -NaN (0xFF) becomes 0xFF80, matching torch's cast.
  result = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(expMan, vNanShifted),
                         Q6_V_vor_VV(sign, vNan), result);

  return result;
}

// Decode 128 packed E4M3FN bytes into 128 fp16 lanes (a vector pair).
// Bias 7, has -0, sole NaN S.1111.111 -> fp16 NaN 0x7F80 with the sign
// PRESERVED.
__attribute__((used)) HVX_VectorPair convert_fp8_fp16_fn(HVX_Vector in) {
  // Widen the packed bytes into halfword lanes; vshuff pre-permutes so the pair
  // comes out in element order.
  HVX_VectorPair hbits_w = Q6_Wuh_vzxt_Vub(Q6_Vb_vshuff_Vb(in));

  return Q6_W_vcombine_VV(decodeFp8HalfToF16Fn(Q6_V_hi_W(hbits_w)),
                          decodeFp8HalfToF16Fn(Q6_V_lo_W(hbits_w)));
}

// Encode one half of the payload: 64 fp16 lanes into 64 E4M3FN codes, each in
// the low byte of its halfword lane. Called once per half so the body is
// written once instead of being mirrored for _lo and _hi.
static inline __attribute__((always_inline)) HVX_Vector
encodeF16HalfToFp8Fn(HVX_Vector in) {
  // Splat constants. always_inline lets both halves share one splat each.
  // Several values serve double duty as shift amounts and comparison operands
  // (e.g. v7 is the mantissa right-shift, the sign left-shift, and the
  // "mantissa == 7" test).
  HVX_Vector v0 = Q6_Vh_vsplat_R(0);
  HVX_Vector v1 = Q6_Vh_vsplat_R(1);
  HVX_Vector v3 = Q6_Vh_vsplat_R(F8_MANT_BITS);
  HVX_Vector v7 = Q6_Vh_vsplat_R(F8_MANT_MAX);
  HVX_Vector v8 = Q6_Vh_vsplat_R(F8_MANT_CARRY);
  HVX_Vector v10 = Q6_Vh_vsplat_R(HF_MANT_BITS);
  HVX_Vector v15 = Q6_Vh_vsplat_R(F8_EXP_MAX);
  HVX_Vector vSignShift = Q6_Vh_vsplat_R(HF_SIGN_SHIFT);
  HVX_Vector vExpMask = Q6_Vh_vsplat_R(HF_EXP_MASK);
  HVX_Vector vMantMask = Q6_Vh_vsplat_R(HF_MANT_MASK);
  HVX_Vector vImplicitOne = Q6_Vh_vsplat_R(HF_IMPLICIT_ONE);
  HVX_Vector vRoundHalf = Q6_Vh_vsplat_R(HF_ROUND_HALF);
  HVX_Vector vRoundMask = Q6_Vh_vsplat_R(F8_NAN_FN); // low 7 dropped bits
  HVX_Vector vRebias = Q6_Vh_vsplat_R(HF_REBIAS_FN);
  HVX_Vector vSubBase = Q6_Vh_vsplat_R(HF_SUB_SHIFT_BASE_FN);
  HVX_Vector vNan = Q6_Vh_vsplat_R(F8_NAN_FN);
  HVX_Vector vMaxFinite = Q6_Vh_vsplat_R(F8_MAX_FINITE_FN);

  // Decompose the fp16 fields and rebias the exponent into fp8's range.
  HVX_Vector sign = Q6_Vh_vlsr_VhVh(in, vSignShift);
  HVX_Vector exp = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(in, v10), vExpMask);
  HVX_Vector mant = Q6_V_vand_VV(in, vMantMask);
  HVX_Vector signif = Q6_V_vor_VV(mant, vImplicitOne);
  HVX_Vector E8 = Q6_Vh_vsub_VhVh(exp, vRebias);

  // ---- Normal candidate (E8 >= 1): keep the top 3 mantissa bits, RNE the
  // low 7.
  HVX_Vector mantTop = Q6_Vh_vlsr_VhVh(mant, v7);
  HVX_Vector roundRem = Q6_V_vand_VV(mant, vRoundMask);
  HVX_VectorPred roundUp_q =
      Q6_Q_or_QQ(Q6_Q_vcmp_gt_VhVh(roundRem, vRoundHalf),
                 Q6_Q_and_QQ(Q6_Q_vcmp_eq_VhVh(roundRem, vRoundHalf),
                             Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(mantTop, v1), v1)));
  HVX_Vector mantTopR =
      Q6_Vh_vadd_VhVh(mantTop, Q6_V_vmux_QVV(roundUp_q, v1, v0));

  // A mantissa that rounds up out of 3 bits carries into the exponent.
  HVX_VectorPred carry_q = Q6_Q_vcmp_eq_VhVh(mantTopR, v8);
  mantTopR = Q6_V_vmux_QVV(carry_q, v0, mantTopR);
  HVX_Vector E8n = Q6_Vh_vadd_VhVh(E8, Q6_V_vmux_QVV(carry_q, v1, v0));
  HVX_Vector normalCode = Q6_V_vor_VV(Q6_Vh_vasl_VhVh(E8n, v3), mantTopR);

  // E4M3FN has no Inf, and its top code (E8n == 15, mantissa == 7) is the NaN
  // slot, so both an exponent overflow and rounding up into that slot
  // saturate to the max finite code (0x7E, +/-448) -- matching torch's
  // saturating cast.
  HVX_VectorPred sat_q =
      Q6_Q_or_QQ(Q6_Q_vcmp_gt_VhVh(E8n, v15),
                 Q6_Q_and_QQ(Q6_Q_vcmp_eq_VhVh(E8n, v15),
                             Q6_Q_vcmp_eq_VhVh(mantTopR, v7)));
  normalCode = Q6_V_vmux_QVV(sat_q, vMaxFinite, normalCode);

  // ---- Subnormal candidate (E8 <= 0): mant8 = signif >> (16 - exp16), RNE.
  HVX_Vector shift = Q6_Vh_vsub_VhVh(vSubBase, exp);
  HVX_Vector mant8 = Q6_Vh_vlsr_VhVh(signif, shift);
  HVX_Vector oneShifted = Q6_Vh_vasl_VhVh(v1, shift);
  HVX_Vector subRem = Q6_V_vand_VV(signif, Q6_Vh_vsub_VhVh(oneShifted, v1));
  HVX_Vector subHalf = Q6_Vh_vlsr_VhVh(oneShifted, v1);
  HVX_VectorPred subUp_q =
      Q6_Q_or_QQ(Q6_Q_vcmp_gt_VhVh(subRem, subHalf),
                 Q6_Q_and_QQ(Q6_Q_vcmp_eq_VhVh(subRem, subHalf),
                             Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(mant8, v1), v1)));
  HVX_Vector mant8R = Q6_Vh_vadd_VhVh(mant8, Q6_V_vmux_QVV(subUp_q, v1, v0));
  // Rounding up out of the subnormal range lands on the minimum normal (0x08).
  HVX_Vector subCode = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(mant8R, v7), v8, mant8R);

  // E8 < 1 selects the subnormal/underflow path.
  HVX_Vector code =
      Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(v1, E8), subCode, normalCode);

  // fp16 zero / subnormal (exp16 == 0) flushes to +/-0. fp16 specials
  // (exp16 == 0x1F): Inf saturates to the max finite code (0x7E); NaN still
  // maps to the NaN code (0x7F).
  code = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(exp, v0), v0, code);
  HVX_VectorPred maxExp_q = Q6_Q_vcmp_eq_VhVh(exp, vExpMask);
  HVX_VectorPred inf_q = Q6_Q_and_QQ(maxExp_q, Q6_Q_vcmp_eq_VhVh(mant, v0));
  HVX_VectorPred nan_q = Q6_Q_and_QQ(maxExp_q, Q6_Q_vcmp_gt_VhVh(mant, v0));
  code = Q6_V_vmux_QVV(inf_q, vMaxFinite, code);
  code = Q6_V_vmux_QVV(nan_q, vNan, code);

  // Apply the sign into bit 7 of the byte code (E4M3FN keeps -0, so unlike the
  // FNUZ encoder there is no zero fixup here).
  HVX_Vector withSign = Q6_V_vor_VV(code, Q6_Vh_vasl_VhVh(sign, v7));

  return withSign;
}

// Encode 128 fp16 lanes (a vector pair) into 128 packed E4M3FN bytes.
// RNE rounding; bias 7 (max finite 448); overflow / Inf saturate to 0x7E
// (sign applied, so -> 0xFE for negatives); NaN -> 0x7F (0xFF for negatives);
// fp16 zero and subnormals -> +/-0.
__attribute__((used)) HVX_Vector
convert_fp16_fp8_fn(HVX_VectorPair in_vhf_pair) {
  // Gather the low byte of each halfword lane, preserving element order.
  return Q6_Vb_vpacke_VhVh(encodeF16HalfToFp8Fn(Q6_V_hi_W(in_vhf_pair)),
                           encodeF16HalfToFp8Fn(Q6_V_lo_W(in_vhf_pair)));
}

#endif // defined(__HVX_ARCH__) && (__HVX_ARCH__ >= 81)

#ifdef __cplusplus
}
#endif
