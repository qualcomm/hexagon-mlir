//===- PrintRuntimeCommon.h - shared tt.print formatting ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Formatting helpers shared by the DSP print shims. The functions are static
// inline and take an emit callback so each translation unit links only against
// its own logger.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_PRINTRUNTIMECOMMON_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_PRINTRUNTIMECOMMON_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Maximum number of tensor elements printed before eliding the remainder.
#define HEXAGON_PRINT_ELEM_CAP 128

typedef void (*hexagon_print_emit_fn)(const char *line);

static inline const char *hexagon_print_prefix(const char *prefix) {
  return prefix ? prefix : "";
}

// Emit `buf`, appending an explicit marker when snprintf reported that the
// formatted text did not fit, so truncation is never silent. `written` is the
// snprintf return value (the length the output would have had).
static inline void hexagon_print_emit_checked(char *buf, size_t bufSize,
                                              int written,
                                              hexagon_print_emit_fn emit) {
  if (written >= (int)bufSize) {
    const char marker[] = "...<truncated>";
    size_t markerLen = sizeof(marker) - 1;
    if (bufSize > markerLen)
      memcpy(buf + bufSize - markerLen - 1, marker, markerLen + 1);
  }
  emit(buf);
}

static inline void hexagon_print_scalar_f32_impl(const char *prefix, int hex,
                                                 float v,
                                                 hexagon_print_emit_fn emit) {
  char buf[128];
  int w;
  (void)hex;
  w = snprintf(buf, sizeof(buf), "%s%f", hexagon_print_prefix(prefix), v);
  hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
}

static inline void hexagon_print_scalar_i32_impl(const char *prefix, int hex,
                                                 int32_t v, int isSigned,
                                                 hexagon_print_emit_fn emit) {
  char buf[128];
  int w;
  if (hex)
    w = snprintf(buf, sizeof(buf), "%s0x%x", hexagon_print_prefix(prefix),
                 (unsigned)v);
  else if (isSigned)
    w = snprintf(buf, sizeof(buf), "%s%d", hexagon_print_prefix(prefix),
                 (int)v);
  else
    w = snprintf(buf, sizeof(buf), "%s%u", hexagon_print_prefix(prefix),
                 (unsigned)v);
  hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
}

static inline void hexagon_print_tensor_f32_impl(const char *prefix, int hex,
                                                 const float *data, int64_t n,
                                                 hexagon_print_emit_fn emit) {
  char buf[256];
  int w;
  int64_t cap, i;
  (void)hex;
  w = snprintf(buf, sizeof(buf), "%s[n=%lld]", hexagon_print_prefix(prefix),
               (long long)n);
  hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  cap = n < HEXAGON_PRINT_ELEM_CAP ? n : HEXAGON_PRINT_ELEM_CAP;
  for (i = 0; i < cap; ++i) {
    w = snprintf(buf, sizeof(buf), "  [%lld] %f", (long long)i, data[i]);
    hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  }
  if (n > cap) {
    w = snprintf(buf, sizeof(buf), "  ... (%lld more)", (long long)(n - cap));
    hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  }
}

static inline void hexagon_print_tensor_i32_impl(const char *prefix, int hex,
                                                 const int32_t *data, int64_t n,
                                                 int isSigned,
                                                 hexagon_print_emit_fn emit) {
  char buf[256];
  int w;
  int64_t cap, i;
  w = snprintf(buf, sizeof(buf), "%s[n=%lld]", hexagon_print_prefix(prefix),
               (long long)n);
  hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  cap = n < HEXAGON_PRINT_ELEM_CAP ? n : HEXAGON_PRINT_ELEM_CAP;
  for (i = 0; i < cap; ++i) {
    if (hex)
      w = snprintf(buf, sizeof(buf), "  [%lld] 0x%x", (long long)i,
                   (unsigned)data[i]);
    else if (isSigned)
      w = snprintf(buf, sizeof(buf), "  [%lld] %d", (long long)i, (int)data[i]);
    else
      w = snprintf(buf, sizeof(buf), "  [%lld] %u", (long long)i,
                   (unsigned)data[i]);
    hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  }
  if (n > cap) {
    w = snprintf(buf, sizeof(buf), "  ... (%lld more)", (long long)(n - cap));
    hexagon_print_emit_checked(buf, sizeof(buf), w, emit);
  }
}

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_PRINTRUNTIMECOMMON_H_
