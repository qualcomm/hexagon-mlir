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
#include "prof_utils.h"
#include "tensor.h"
#include "test_report.h"
#include <HAP_farf.h>
#include <cstdint>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

const int BLOCK_SIZE = 63 * 16384;

extern "C" void add_2d_mlir(int64_t, MemRefDescriptor<float, 2> *, int64_t,
                            MemRefDescriptor<float, 2> *, int64_t,
                            MemRefDescriptor<float, 2> *, int, int, int, int,
                            int, int);

bool check(float *A, float *B, float *C) {
  for (int i = 0; i < BLOCK_SIZE; i++) {
    float diff = A[i] + B[i] - C[i];
    if (diff > 1e-3 || diff < -1e-3) {
      FARF(ALWAYS, "Mismatch at index %d\n%f + %f == %f (got %f, diff %f)\n", i,
           A[i], B[i], A[i] + B[i], C[i], diff);
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
  for (size_t i = 0; i < BLOCK_SIZE; i++)
    ptr[i] = dist(gen);
}

int main() {
  Tensor<float, 2> T0({{63, 16384}, {16384, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 2> *dt0 = T0.toMemRefDesc();
  Tensor<float, 2> T1({{63, 16384}, {16384, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 2> *dt1 = T1.toMemRefDesc();
  Tensor<float, 2> T2({{63, 16384}, {16384, 1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 2> *dt2 = T2.toMemRefDesc();

  set_rand(dt0->allocated);
  set_rand(dt1->allocated);
  set_rand(dt2->allocated);

  uint64_t avg_cycles = benchmark_time_us(
      1, [&]() { add_2d_mlir(2, dt0, 2, dt1, 2, dt2, 0, 0, 0, 0, 0, 0); });

  Result result = Result::Pass;
  if (!check(dt0->allocated, dt1->allocated, dt2->allocated))
    result = Result::Fail;
  WriteLWPOutput("/vendor/bin/lwp.json");
  TestReport tr("add_2d_mlir", avg_cycles, "microseconds", result);
  tr.save();

  return 0;
}
