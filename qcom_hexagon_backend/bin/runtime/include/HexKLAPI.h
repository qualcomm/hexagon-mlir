//===- HexKLAPI.h - HExKL C runtime call API   -----------------------------==//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H

#include "HAP_power.h"
#include <memory>
#include <stdint.h>

extern "C" {
int hexkl_matmul_f16f16_f32(int64_t n_row, int64_t n_col, int64_t n_inner,
                            float *restrict outM,
                            const _Float16 *restrict inAct,
                            const _Float16 *restrict inW);
}
#endif // # HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H
