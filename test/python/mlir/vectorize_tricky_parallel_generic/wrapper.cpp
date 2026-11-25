//===-- wrapper.cpp -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <random>

#define FARF_ALWAYS 1
#include "CRunnerUtils.cpp"
#include "common.h"
#include "hexagon_benchmark.h"
#include "tensor.h"
#include "test_report.h"

#include <HAP_farf.h>
#include <cstdint>

static constexpr int D0 = 2;
static constexpr int D1 = 24;
static constexpr int D2 = 8;
static constexpr int D3 = 2;
static constexpr int D4 = 32;
static constexpr int D5 = 2;

static constexpr _Float16 kDivisor = (_Float16)256.0f;
static constexpr _Float16 kAddScalar = (_Float16)1.0e-5f;

extern "C" void vectorize_tricky_parallel_generic_mlir(
    int64_t, MemRefDescriptor<_Float16, 6> *, int64_t,
    MemRefDescriptor<_Float16, 4> *, int64_t, MemRefDescriptor<_Float16, 2> *,
    int64_t, MemRefDescriptor<_Float16, 2> *, int64_t,
    MemRefDescriptor<_Float16, 6> *, int, int, int, int, int, int, int, int);

static void fill_uniform(_Float16 *ptr, size_t n, float lo, float hi) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < n; ++i)
    ptr[i] = (_Float16)dist(gen);
}

static bool check(const _Float16 *in, const _Float16 *in136,
                  const _Float16 *in137, const _Float16 *in138,
                  const _Float16 *out) {
  float maxAbsDiff = 0.0f;
  for (int d0 = 0; d0 < D0; ++d0) {
    for (int d1 = 0; d1 < D1; ++d1) {
      for (int d2 = 0; d2 < D2; ++d2) {
        for (int d3 = 0; d3 < D3; ++d3) {
          for (int d4 = 0; d4 < D4; ++d4) {
            for (int d5 = 0; d5 < D5; ++d5) {
              int idx =
                  (((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * D5 +
                   d5);
              int idx136 = (((d0 * D2 + d2) * D3 + d3) * D5 + d5);
              int idx2 = d1 * D4 + d4;

              _Float16 t0 = (_Float16)(in136[idx136] / kDivisor);
              _Float16 t1 = (_Float16)(t0 + kAddScalar);
              _Float16 t2 = (_Float16)(1.0f / std::sqrt((float)t1));
              _Float16 t3 = (_Float16)(in[idx] * t2);
              _Float16 t4 = (_Float16)(t3 * in137[idx2]);
              _Float16 ref = (_Float16)(t4 + in138[idx2]);

              float diff = (float)ref - (float)out[idx];
              float absDiff = std::abs(diff);
              if (absDiff > maxAbsDiff)
                maxAbsDiff = absDiff;
              if (diff > 0.1f || diff < -0.1f) {
                printf(
                    "Mismatch at (%d,%d,%d,%d,%d,%d) ref=%f got=%f diff=%f\n",
                    d0, d1, d2, d3, d4, d5, (float)ref, (float)out[idx], diff);
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

int main() {
  Tensor<_Float16, 6> Tin({{D0, D1, D2, D3, D4, D5},
                           {D1 * D2 * D3 * D4 * D5, D2 * D3 * D4 * D5,
                            D3 * D4 * D5, D4 * D5, D5, 1},
                           128},
                          MemType::HEAP);
  Tensor<_Float16, 4> Tin136(
      {{D0, D2, D3, D5}, {D2 * D3 * D5, D3 * D5, D5, 1}, 128}, MemType::HEAP);
  Tensor<_Float16, 2> Tin137({{D1, D4}, {D4, 1}, 128}, MemType::HEAP);
  Tensor<_Float16, 2> Tin138({{D1, D4}, {D4, 1}, 128}, MemType::HEAP);
  Tensor<_Float16, 6> Tout({{D0, D1, D2, D3, D4, D5},
                            {D1 * D2 * D3 * D4 * D5, D2 * D3 * D4 * D5,
                             D3 * D4 * D5, D4 * D5, D5, 1},
                            128},
                           MemType::HEAP);

  auto *din = Tin.toMemRefDesc();
  auto *din136 = Tin136.toMemRefDesc();
  auto *din137 = Tin137.toMemRefDesc();
  auto *din138 = Tin138.toMemRefDesc();
  auto *dout = Tout.toMemRefDesc();

  fill_uniform(din->allocated, static_cast<size_t>(D0) * D1 * D2 * D3 * D4 * D5,
               -1.0f, 1.0f);
  fill_uniform(din136->allocated, static_cast<size_t>(D0) * D2 * D3 * D5, 1.0f,
               16.0f);
  fill_uniform(din137->allocated, static_cast<size_t>(D1) * D4, -1.0f, 1.0f);
  fill_uniform(din138->allocated, static_cast<size_t>(D1) * D4, -1.0f, 1.0f);
  fill_uniform(dout->allocated,
               static_cast<size_t>(D0) * D1 * D2 * D3 * D4 * D5, -1.0f, 1.0f);

  uint64_t avg_cycles = benchmark_time_us(1, [&]() {
    vectorize_tricky_parallel_generic_mlir(6, din, 4, din136, 2, din137, 2,
                                           din138, 6, dout, 0, 0, 0, 0, 0, 0, 0,
                                           0);
  });

  Result result = Result::Pass;
  if (!check(din->allocated, din136->allocated, din137->allocated,
             din138->allocated, dout->allocated))
    result = Result::Fail;

  TestReport tr("vectorize_tricky_parallel_generic_mlir", avg_cycles,
                "microseconds", result);
  tr.save();
  return 0;
}
