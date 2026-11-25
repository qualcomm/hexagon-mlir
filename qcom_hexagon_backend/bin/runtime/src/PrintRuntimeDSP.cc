//===- PrintRuntimeDSP.cc - tt.print runtime shims (DSP / FARF) -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// DSP-path implementation of the tt.print (tl.device_print) runtime shims.
// Emits via FARF.
//
//===----------------------------------------------------------------------===//

#include "PrintRuntimeCommon.h"

#if defined(__hexagon__)
#ifndef FARF_ALWAYS
#define FARF_ALWAYS 1
#endif
#include <HAP_farf.h>
#endif

static void hexagon_print_emit_dsp(const char *line) {
#if defined(__hexagon__)
  FARF(ALWAYS, "%s", line);
#else
  printf("%s\n", line);
#endif
}

extern "C" {

void hexagon_print_scalar_f32_dsp(const char *prefix, int hex, float v) {
  hexagon_print_scalar_f32_impl(prefix, hex, v, hexagon_print_emit_dsp);
}

void hexagon_print_scalar_i32_dsp(const char *prefix, int hex, int32_t v,
                                  int isSigned) {
  hexagon_print_scalar_i32_impl(prefix, hex, v, isSigned,
                                hexagon_print_emit_dsp);
}

void hexagon_print_tensor_f32_dsp(const char *prefix, int hex,
                                  const float *data, int64_t n) {
  hexagon_print_tensor_f32_impl(prefix, hex, data, n, hexagon_print_emit_dsp);
}

void hexagon_print_tensor_i32_dsp(const char *prefix, int hex,
                                  const int32_t *data, int64_t n,
                                  int isSigned) {
  hexagon_print_tensor_i32_impl(prefix, hex, data, n, isSigned,
                                hexagon_print_emit_dsp);
}
}
