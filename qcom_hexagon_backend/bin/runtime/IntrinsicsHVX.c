//===- IntrinsicsHVX.c   - some useful HVX qfloat routines ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Contains useful HVX qfloat routines.
// All routines should be tagged with HEXAGON_INTRIN or HEXAGON_INTRIN_INLINE
// See llvm/utils/Target/Hexagon/h*_protos.h for available intrinsics
//
//===----------------------------------------------------------------------===//

#include "hexagon_types.h"
#include "internal/hvx_internal.h"
#include "internal/qhmath_hvx_convert.h"
#include "internal/qhmath_hvx_vector.h"

#ifndef HEXAGON_INTRIN_INLINE
#define HEXAGON_INTRIN_INLINE                                                  \
  __attribute__((unused, used, always_inline, visibility("hidden")))
#endif
#ifndef HEXAGON_INTRIN
#define HEXAGON_INTRIN __attribute__((visibility("hidden")))
#endif

#ifndef MAP_MLIR_TO_QHMATH_DOUBLE_ARGS
#define MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(FNAME, SUFFIX)                          \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(       \
      HVX_Vector vin1, HVX_Vector vin2) {                                      \
    return qhmath_hvx_##FNAME##_##SUFFIX(vin1, vin2);                          \
  }
#endif

/* Detect NaN via unsigned integer comparison: a value is NaN iff
 * (bits & 0x7FFFFFFF) > 0x7F800000 (fp32) or (bits & 0x7FFF) > 0x7C00 (fp16).
 * Uses ungated intrinsics available from v68 onwards. */
#define MAP_MLIR_TO_QHMATH_SINGLE_ARG_vf(FNAME)                                \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__vf(             \
      HVX_Vector vin) {                                                        \
    HVX_Vector result = qhmath_hvx_##FNAME##_vf(vin);                          \
    HVX_Vector abs_mask = Q6_V_vsplat_R(0x7FFFFFFF);                           \
    HVX_Vector inf_bits = Q6_V_vsplat_R(0x7F800000);                           \
    HVX_Vector abs_vin = Q6_V_vand_VV(vin, abs_mask);                          \
    HVX_VectorPred is_nan = Q6_Q_vcmp_gt_VuwVuw(abs_vin, inf_bits);            \
    return Q6_V_vmux_QVV(is_nan, vin, result);                                 \
  }
#define MAP_MLIR_TO_QHMATH_SINGLE_ARG_vhf(FNAME)                               \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__vhf(            \
      HVX_Vector vin) {                                                        \
    HVX_Vector result = qhmath_hvx_##FNAME##_vhf(vin);                         \
    HVX_Vector abs_mask = Q6_V_vsplat_R(0x7FFF7FFF);                           \
    HVX_Vector inf_bits = Q6_V_vsplat_R(0x7C007C00);                           \
    HVX_Vector abs_vin = Q6_V_vand_VV(vin, abs_mask);                          \
    HVX_VectorPred is_nan = Q6_Q_vcmp_gt_VuhVuh(abs_vin, inf_bits);            \
    return Q6_V_vmux_QVV(is_nan, vin, result);                                 \
  }
#ifndef MAP_MLIR_TO_QHMATH_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_SINGLE_ARG(FNAME, SUFFIX)                           \
  MAP_MLIR_TO_QHMATH_SINGLE_ARG_##SUFFIX(FNAME)
#endif

// The following macro is necessary for internal functions, such as
// floor and ceil, which have different naming conventions than the
// qhmath functions exposed publicly. The core difference is the
// having the SUFFIX appear twice in the function name.
#ifndef MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(FNAME, SUFFIX)                  \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(       \
      HVX_Vector vin) {                                                        \
    return qhmath_hvx_##SUFFIX##_##FNAME##_##SUFFIX(vin);                      \
  }
#endif

//===----------------------------------------------------------------------===//
// Exponential
//===----------------------------------------------------------------------===//

/* exp(x): qhmath_hvx_exp_vf returns garbage finite values (not NaN) for
 * overflow (x > ~88.72) where IEEE-754 requires +INF, and ALSO for
 * underflow (x <= ~-88.72, including x = -inf, e.g. masked-softmax lanes)
 * where IEEE-754 requires +0.0 — the polynomial's own negative-exponent
 * zeroing guard does not catch every underflow input reliably (observed:
 * exp(-inf) returns FLT_MAX instead of 0.0). Detect both from the input
 * side: if vin > overflow_threshold, replace result with +inf; if
 * vin < -overflow_threshold (covers -inf), replace result with +0.0.
 * If input was NaN, propagate it unchanged.
 *
 * Q6_Q_vcmp_gt_VsfVsf is ungated (available on v68+); NaN inputs compare
 * false so they do not trigger the overflow/underflow branches. */
#define _HVX_EXP_OVERFLOW_SAT_VF(result, vin)                                 \
  do {                                                                         \
    HVX_Vector _abs_mask = Q6_V_vsplat_R(0x7FFFFFFF);                          \
    HVX_Vector _inf_bits = Q6_V_vsplat_R(0x7F800000);                          \
    HVX_Vector _zero_bits = Q6_V_vzero();                                      \
    /* ln(FLT_MAX) in fp32 bits */                                             \
    HVX_Vector _thresh = Q6_V_vsplat_R(0x42B17218);                            \
    /* -ln(FLT_MAX) in fp32 bits */                                            \
    HVX_Vector _neg_thresh = Q6_V_vsplat_R(0xC2B17218);                        \
    HVX_Vector _abs_vin = Q6_V_vand_VV((vin), _abs_mask);                      \
    HVX_VectorPred _in_nan = Q6_Q_vcmp_gt_VuwVuw(_abs_vin, _inf_bits);         \
    HVX_VectorPred _overflow = Q6_Q_vcmp_gt_VsfVsf((vin), _thresh);            \
    HVX_VectorPred _underflow = Q6_Q_vcmp_gt_VsfVsf(_neg_thresh, (vin));       \
    HVX_Vector _step1 = Q6_V_vmux_QVV(_underflow, _zero_bits, (result));       \
    HVX_Vector _step2 = Q6_V_vmux_QVV(_overflow, _inf_bits, _step1);           \
    (result) = Q6_V_vmux_QVV(_in_nan, (vin), _step2);                          \
  } while (0)

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_exp__vf(HVX_Vector vin) {
  HVX_Vector result = qhmath_hvx_exp_vf(vin);
  _HVX_EXP_OVERFLOW_SAT_VF(result, vin);
  return result;
}

/* IEEE-safe array wrapper used by libdevice.exp (tl.exp fp32).
 * Same ABI as qhmath_hvx_exp_af: int32_t(float*, float*, uint32_t). */
HEXAGON_INTRIN int32_t qhmath_hvx_exp_af_ieee(float *restrict input,
                                              float *restrict output,
                                              uint32_t size) {
  if (!input || !output || !size)
    return -1;
  int32_t leftover = size & 31;
  int32_t leftover_size = leftover * sizeof(float);
  int32_t nvecs = size / 32;
  HVX_Vector *ip = (HVX_Vector *)input;
  HVX_UVector *op = (HVX_UVector *)output;
  HVX_Vector slinep = *ip++;
  for (int32_t i = nvecs - 1; i > 0; --i) {
    HVX_Vector slinec = *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vf(sline);
    _HVX_EXP_OVERFLOW_SAT_VF(result, sline);
    *op++ = (HVX_UVector)result;
    slinep = slinec;
  }
  if (nvecs > 0) {
    HVX_Vector slinec =
        (is_aligned(ip, VLEN) && leftover == 0) ? slinep : *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vf(sline);
    _HVX_EXP_OVERFLOW_SAT_VF(result, sline);
    *op++ = (HVX_UVector)result;
    slinep = slinec;
  }
  if (leftover > 0) {
    HVX_Vector slinec =
        is_in_one_chunk(ip, leftover_size, VLEN) ? slinep : *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vf(sline);
    _HVX_EXP_OVERFLOW_SAT_VF(result, sline);
    vstu_variable(op, leftover_size, result);
  }
  return 0;
}

/* Same as _HVX_EXP_OVERFLOW_SAT_VF but for fp16.
 * Q6_Q_vcmp_gt_VhfVhf is ungated (available on v68+). */
#define _HVX_EXP_OVERFLOW_SAT_VHF(result, vin)                                \
  do {                                                                         \
    HVX_Vector _abs_mask = Q6_V_vsplat_R(0x7FFF7FFF);                          \
    HVX_Vector _inf_bits = Q6_V_vsplat_R(0x7C007C00);                          \
    HVX_Vector _zero_bits = Q6_V_vzero();                                      \
    /* ln(fp16_max=65504) ≈ 11.0898, fp16 bits 0x498B packed */              \
    HVX_Vector _thresh = Q6_V_vsplat_R(0x498B498B);                            \
    /* -ln(fp16_max), fp16 bits 0xC98B packed */                               \
    HVX_Vector _neg_thresh = Q6_V_vsplat_R(0xC98BC98B);                        \
    HVX_Vector _abs_vin = Q6_V_vand_VV((vin), _abs_mask);                      \
    HVX_VectorPred _in_nan = Q6_Q_vcmp_gt_VuhVuh(_abs_vin, _inf_bits);         \
    HVX_VectorPred _overflow = Q6_Q_vcmp_gt_VhfVhf((vin), _thresh);            \
    HVX_VectorPred _underflow = Q6_Q_vcmp_gt_VhfVhf(_neg_thresh, (vin));       \
    HVX_Vector _step1 = Q6_V_vmux_QVV(_underflow, _zero_bits, (result));       \
    HVX_Vector _step2 = Q6_V_vmux_QVV(_overflow, _inf_bits, _step1);           \
    (result) = Q6_V_vmux_QVV(_in_nan, (vin), _step2);                          \
  } while (0)

/* Same overflow saturation for fp16 exp. */
HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_exp__vhf(HVX_Vector vin) {
  HVX_Vector result = qhmath_hvx_exp_vhf(vin);
  _HVX_EXP_OVERFLOW_SAT_VHF(result, vin);
  return result;
}

/* IEEE-safe array wrapper for fp16 used by libdevice.exp (tl.exp fp16).
 * Same ABI as qhmath_hvx_exp_ahf: int32_t(fp16*, fp16*, uint32_t). */
HEXAGON_INTRIN int32_t qhmath_hvx_exp_ahf_ieee(__fp16 *restrict input,
                                               __fp16 *restrict output,
                                               uint32_t size) {
  if (!input || !output || !size)
    return -1;
  int32_t leftover = size & 63;
  int32_t leftover_size = leftover * sizeof(__fp16);
  int32_t nvecs = size / 64;
  HVX_Vector *ip = (HVX_Vector *)input;
  HVX_UVector *op = (HVX_UVector *)output;
  HVX_Vector slinep = *ip++;
  for (int32_t i = nvecs - 1; i > 0; --i) {
    HVX_Vector slinec = *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vhf(sline);
    _HVX_EXP_OVERFLOW_SAT_VHF(result, sline);
    *op++ = (HVX_UVector)result;
    slinep = slinec;
  }
  if (nvecs > 0) {
    HVX_Vector slinec =
        (is_aligned(ip, VLEN) && leftover == 0) ? slinep : *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vhf(sline);
    _HVX_EXP_OVERFLOW_SAT_VHF(result, sline);
    *op++ = (HVX_UVector)result;
    slinep = slinec;
  }
  if (leftover > 0) {
    HVX_Vector slinec =
        is_in_one_chunk(ip, leftover_size, VLEN) ? slinep : *ip++;
    HVX_Vector sline = Q6_V_valign_VVR(slinec, slinep, (size_t)input);
    HVX_Vector result = qhmath_hvx_exp_vhf(sline);
    _HVX_EXP_OVERFLOW_SAT_VHF(result, sline);
    vstu_variable(op, leftover_size, result);
  }
  return 0;
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_floor__vf(HVX_Vector vin) {
  return qhmath_hvx_vsf_floor_vsf(vin);
}

#undef _HVX_EXP_OVERFLOW_SAT_VF
#undef _HVX_EXP_OVERFLOW_SAT_VHF

//===----------------------------------------------------------------------===//
// Power
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(pow, vf);
MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(pow, vhf);

//===----------------------------------------------------------------------===//
// Trigonometric Functions
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_SINGLE_ARG(sin, vf);
// MAP_MLIR_TO_QHMATH_SINGLE_ARG(sin, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(cos, vf);
// MAP_MLIR_TO_QHMATH_SINGLE_ARG(cos, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(tan, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(tanh, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(tanh, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(acos, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(acos, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(asin, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(asin, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(atan, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(atan, vhf);

//===----------------------------------------------------------------------===//
// Sqrt
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_SINGLE_ARG(sqrt, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(sqrt, vhf);

//===----------------------------------------------------------------------===//
// Rsqrt
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_SINGLE_ARG(rsqrt, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(rsqrt, vhf);

//===----------------------------------------------------------------------===//
// Floor
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(floor, vhf);
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(floor, vsf);

//===----------------------------------------------------------------------===//
// Ceil
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(ceil, vhf);
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(ceil, vsf);

#undef MAP_MLIR_TO_QHMATH_SINGLE_ARG_vf
#undef MAP_MLIR_TO_QHMATH_SINGLE_ARG_vhf
