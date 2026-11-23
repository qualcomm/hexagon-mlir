//===- wrapper.cpp --------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <random>

#define FARF_ALWAYS 1
#include "CRunnerUtils.cpp"
#include "HAP_farf.h"
#include "common.h"
#include "hexagon_benchmark.h"
#include "tensor.h"
#include "test_report.h"

static const int D0 = 2, D1 = 24, D2 = 8, D3 = 2, D4 = 32, D5 = 2;

extern "C" void vectorize_sum_of_squares_mlir(
    int64_t, MemRefDescriptor<_Float16, 6> *, // in0 [D0,D1,D2,D3,D4,D5]
    int64_t, MemRefDescriptor<_Float16, 6> *, // in1 [D0,D1,D2,D3,D4,D5]
    int64_t, MemRefDescriptor<_Float16, 4> *, // out [D0,D2,D3,D5]
    int, int, int, int, int, int, int);

static bool check(const _Float16 *in0, const _Float16 *in1,
                  const _Float16 *out) {
  bool ok = true;
  for (int d0 = 0; d0 < D0; ++d0)
    for (int d2 = 0; d2 < D2; ++d2)
      for (int d3 = 0; d3 < D3; ++d3)
        for (int d5 = 0; d5 < D5; ++d5) {
          _Float16 ref = (_Float16)0.0f;
          for (int d1 = 0; d1 < D1; ++d1)
            for (int d4 = 0; d4 < D4; ++d4) {
              int idx =
                  (((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * D5 +
                   d5);
              ref = (_Float16)(ref + (_Float16)(in0[idx] * in1[idx]));
            }
          int ridx = (((d0 * D2 + d2) * D3 + d3) * D5 + d5);
          float diff = (float)ref - (float)out[ridx];
          FARF(ALWAYS, "sos[%d,%d,%d,%d] ref=%f got=%f diff=%f", d0, d2, d3, d5,
               (float)ref, (float)out[ridx], diff);
          if (std::abs(diff) > 4.0f)
            ok = false;
        }
  return ok;
}

static void fill_uniform(_Float16 *ptr, size_t n, float lo, float hi) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < n; ++i)
    ptr[i] = (_Float16)dist(gen);
}

int main() {
  Tensor<_Float16, 6> Tin0({{D0, D1, D2, D3, D4, D5},
                            {D1 * D2 * D3 * D4 * D5, D2 * D3 * D4 * D5,
                             D3 * D4 * D5, D4 * D5, D5, 1},
                            128},
                           MemType::HEAP);
  Tensor<_Float16, 6> Tin1({{D0, D1, D2, D3, D4, D5},
                            {D1 * D2 * D3 * D4 * D5, D2 * D3 * D4 * D5,
                             D3 * D4 * D5, D4 * D5, D5, 1},
                            128},
                           MemType::HEAP);
  Tensor<_Float16, 4> Tout(
      {{D0, D2, D3, D5}, {D2 * D3 * D5, D3 * D5, D5, 1}, 128}, MemType::HEAP);

  auto *din0 = Tin0.toMemRefDesc();
  auto *din1 = Tin1.toMemRefDesc();
  auto *dout = Tout.toMemRefDesc();

  fill_uniform(din0->allocated, D0 * D1 * D2 * D3 * D4 * D5, -1.0f, 1.0f);
  fill_uniform(din1->allocated, D0 * D1 * D2 * D3 * D4 * D5, -1.0f, 1.0f);
  // Output is zero-initialized by linalg.fill in the kernel
  for (int i = 0; i < D0 * D2 * D3 * D5; ++i)
    dout->allocated[i] = (_Float16)0.0f;

  double avg_cycles = benchmark_time_us(1, [&]() {
    vectorize_sum_of_squares_mlir(6, din0, 6, din1, 4, dout, 0, 0, 0, 0, 0, 0,
                                  0);
  });

  Result result = Result::Pass;
  if (!check(din0->allocated, din1->allocated, dout->allocated))
    result = Result::Fail;

  TestReport tr("vectorize_sum_of_squares_mlir", avg_cycles, "microseconds",
                result);
  tr.save();
  return 0;
}
