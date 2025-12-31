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
#include <limits>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMS 1024 * 1024

extern "C" void exp_log_kernel(int64_t, MemRefDescriptor<float, 1> *, int64_t,
                               MemRefDescriptor<float, 1> *);

bool check(float *X, float *Y) {
  for (int i = 0; i < NUM_ELEMS; i++) {
    float diff = log(exp(X[i])) - Y[i];
    if (diff > 1e-3 || diff < -1e-3) {
      FARF(ALWAYS, "Mismatch at index %d\nX[i]=%f, Y[i]=%f, diff=%f)\n", i,
           X[i], Y[i], diff);
      return false;
    }
  }
  return true;
}

void set_rand(float *ptr) {
  const float std = 0.5f;
  const float mean = 0.0f;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  for (size_t i = 0; i < NUM_ELEMS; i++)
    ptr[i] = dist(gen);
}

int main() {
  Tensor<float, 1> T0({{NUM_ELEMS}, {1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 1> *dt0 = T0.toMemRefDesc();
  Tensor<float, 1> T1({{NUM_ELEMS}, {1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 1> *dt1 = T1.toMemRefDesc();

  set_rand(dt0->allocated);
  uint64_t avg_cycles =
      benchmark_time_us(1, [&]() { exp_log_kernel(1, dt0, 1, dt1); });

  Result result = Result::Pass;
  if (!check(dt0->allocated, dt1->allocated))
    result = Result::Fail;

  TestReport tr("exp_log_base_mlir", avg_cycles, "microsecond", result);
  tr.save();
  return 0;
}
