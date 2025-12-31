//===-- wrapper.cpp -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "CRunnerUtils.cpp"
#include "common.h"
#include "tensor.h"
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#define FARF_ALWAYS 1
#include "hexagon_benchmark.h"
#include "test_report.h"
#include <HAP_farf.h>
#include <inttypes.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string.h>

extern "C" void conv_tiling_hmx(_Float16 *, _Float16 *, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, _Float16 *, _Float16 *,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, _Float16 *,
                                _Float16 *, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t);

int check(_Float16 *in, _Float16 *filter, _Float16 *out, int in_height,
          int in_width, int out_height, int out_width, int channels,
          int no_filters) {
  int errcnt = 0, maxerr = 10000;
  FARF(ALWAYS, "Checking...\n");
  for (int n = 0; n < no_filters; n++) {
    FARF(ALWAYS, "Checking depth=%d\n", n);
    for (int h = 0; h < out_height; h++) {
      for (int w = 0; w < out_width; w++) {
        float sum = 0;
        for (int c = 0; c < channels; c++) {
          int w_cood = w;
          int h_cood = h;
          if (!(w_cood < 0 || h_cood < 0 || w_cood >= in_width ||
                h_cood >= in_height)) {
            _Float16 activation_val =
                in[c + w_cood * channels + h_cood * in_width * channels];
            _Float16 filter_val = filter[c + n * channels];
            sum += (float)activation_val * (float)filter_val;
          }
        }
        _Float16 computed =
            out[n + w * no_filters + h * out_width * no_filters];
        if (abs((float)computed - sum) > 1e-1) {
          errcnt++;
          if (errcnt <= maxerr) {
            FARF(ALWAYS,
                 "Mismatch at (%3d, %3d, %3d): %f (MLIR) == %f (Expected)\n", n,
                 h, w, (float)computed, (float)sum);
          }
        }
      }
    }
    FARF(ALWAYS, "Depth %d passed.\n", n);
  }
  if (errcnt > maxerr) {
    FARF(ALWAYS, "...\n");
  }
  if (errcnt > 0) {
    FARF(ALWAYS, "Mismatch at %d places\n", errcnt);
  }
  return errcnt;
}

void set_rand(_Float16 *ptr, uint32_t len) {
  for (size_t i = 0; i < len; i++) {
    float val = rand() / (float)RAND_MAX;
    ptr[i] = val;
  }
}
void set_one(_Float16 *ptr, uint32_t len) {
  for (size_t i = 0; i < len; i++) {
    ptr[i] = (_Float16)1;
  }
}
template <typename T> void set_zero(T *ptr, uint32_t len) {
  for (size_t i = 0; i < len; i++)
    ptr[i] = (_Float16)0;
}

int main(int argc, char **argv) {

  std::string bname = "conv_tiling_hmx";

  uint32_t width = 32;
  uint32_t height = 16;
  uint32_t depth = 128;
  uint32_t filters = 224;

  const MemType memType = MemType::HEAP;
  Tensor<_Float16, 4> in({{1, height, width, depth},
                          {width * depth, width * depth, depth, 1},
                          2048},
                         memType);
  Tensor<_Float16, 4> filter(
      {{filters, 1, 1, depth}, {depth, depth, depth, 1}, 2048}, memType);
  Tensor<_Float16, 4> out({{1, height, width, filters},
                           {width * filters, width * filters, filters, 1},
                           2048},
                          memType);

  MemRefDescriptor<_Float16, 4> *dt0 = in.toMemRefDesc();
  MemRefDescriptor<_Float16, 4> *dt1 = filter.toMemRefDesc();
  MemRefDescriptor<_Float16, 4> *dt2 = out.toMemRefDesc();

  set_one(dt0->allocated, height * width * depth);
  set_one(dt1->allocated, filters * depth);
  set_zero(dt2->allocated, height * width * filters);

  uint64_t avg_cycles = benchmark_time_us(50, [&]() {
    conv_tiling_hmx(dt0->allocated, dt0->allocated, 0, 1, height, width, depth,
                    height * width * depth, width * depth, depth, 1,
                    dt1->allocated, dt1->allocated, 0, filters, 1, 1, depth,
                    depth, depth, depth, 1, dt2->allocated, dt2->allocated, 0,
                    1, height, width, filters, height * width * filters,
                    width * filters, filters, 1);
  });
  FARF(ALWAYS, "Time: %f %" PRIu64 "\n", (float)avg_cycles, avg_cycles);

  Result result = Result::Pass;
  int mismatches = check(dt0->allocated, dt1->allocated, dt2->allocated, height,
                         width, height, width, depth, filters);
  if (mismatches) {
    result = Result::Fail;
    FARF(ALWAYS, "Failed");
  }

  TestReport tr(bname, avg_cycles, "microseconds", result);
  tr.save();
  return 0;
}
