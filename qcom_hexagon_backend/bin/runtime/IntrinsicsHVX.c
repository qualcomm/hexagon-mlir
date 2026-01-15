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
  inline __attribute__((unused, used, always_inline, visibility("hidden")))
#endif
#ifndef HEXAGON_INTRIN
#define HEXAGON_INTRIN __attribute__((visibility("hidden")))
#endif

#ifndef MAP_MLIR_TO_QHMATH_DOUBLE_ARGS
#define MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(FNAME, SUFFIX)                          \
  HEXAGON_INTRIN HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(              \
      HVX_Vector vin1, HVX_Vector vin2) {                                      \
    return qhmath_hvx_##FNAME##_##SUFFIX(vin1, vin2);                          \
  }
#endif

#ifndef MAP_MLIR_TO_QHMATH_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_SINGLE_ARG(FNAME, SUFFIX)                           \
  HEXAGON_INTRIN HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(              \
      HVX_Vector vin) {                                                        \
    return qhmath_hvx_##FNAME##_##SUFFIX(vin);                                 \
  }
#endif

// The following macro is necessary for internal functions, such as
// floor and ceil, which have different naming conventions than the
// qhmath functions exposed publicly. The core difference is the
// having the SUFFIX appear twice in the function name.
#ifndef MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(FNAME, SUFFIX)                  \
  HEXAGON_INTRIN HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(              \
      HVX_Vector vin) {                                                        \
    return qhmath_hvx_##SUFFIX##_##FNAME##_##SUFFIX(vin);                      \
  }
#endif

//===----------------------------------------------------------------------===//
// Exponential
//===----------------------------------------------------------------------===//

MAP_MLIR_TO_QHMATH_SINGLE_ARG(exp, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(exp, vhf);

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
