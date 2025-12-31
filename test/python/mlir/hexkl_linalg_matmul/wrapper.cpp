//===-- wrapper.cpp -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#define FARF_ALWAYS 1
#include "CRunnerUtils.cpp"
#include "common.h"
#include "debug.h"
#include "hexagon_benchmark.h"
#include "prof_utils.h"
#include "tensor.h"
#include "test_report.h"
#include <HAP_farf.h>
#include <cstdint>
#include <limits>
#include <random>
#include <stdio.h>
#include <stdlib.h>

extern "C" void hexkl_linalg_matmul(int64_t, MemRefDescriptor<_Float16, 2> *,
                                    int64_t, MemRefDescriptor<_Float16, 2> *,
                                    int64_t, MemRefDescriptor<float, 2> *, int,
                                    int, int, int, int, int);

#define MAX_DIFF 1e-2
constexpr uint32_t N = 1024;
constexpr uint32_t K = 64;
constexpr uint32_t M = 512;

void set_pointer_random(_Float16 *ptr, size_t num_elements) {
  const float std = 0.5f;
  const float mean = 0.0f;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  for (size_t i = 0; i < num_elements; i++)
    ptr[i] = _Float16(roundf(dist(gen) * 10000) /
                      10000); // round off to 4 decimal places
}

// Reference C++ Matmul Implementation from the HexKL library
bool check(float *restrict outM, const _Float16 *restrict inAct,
           const _Float16 *restrict inW) {
  for (uint32_t row = 0; row < N; row++) {
    for (uint32_t col = 0; col < M; col++) {
      float acc = 0;
      for (uint32_t entry = 0; entry < K; entry++) {
        acc += (float)inAct[row * K + entry] * inW[entry * M + col];
      }
      auto outIdx = row * M + col;
      float diff = outM[outIdx] - acc;
      if (diff > MAX_DIFF || diff < -MAX_DIFF) {
        FARF(ALWAYS,
             "Mismatch at index %d, CPU output was %f and Matmul output was "
             "%f\n",
             outIdx, acc, outM[outIdx]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  Tensor<_Float16, 2> T0({{N, K}, {K, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<_Float16, 2> *dt0 = T0.toMemRefDesc();
  Tensor<_Float16, 2> T1({{K, M}, {M, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<_Float16, 2> *dt1 = T1.toMemRefDesc();
  Tensor<float, 2> T2({{N, M}, {M, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 2> *dt2 = T2.toMemRefDesc();

  set_pointer_random(dt0->allocated, N * K);
  set_pointer_random(dt1->allocated, K * M);

  uint64_t avg_time_us = benchmark_time_us(1, [&]() {
    hexkl_linalg_matmul(2, dt0, 2, dt1, 2, dt2, 1, 1, 1, 0, 0, 0);
  });

  Result result = Result::Pass;
  if (!check(dt2->allocated, dt0->allocated, dt1->allocated))
    result = Result::Fail;
  TestReport tr("hexkl_linalg_matmul", avg_time_us, "microseconds", result);
  tr.save();
  return result;
}
