//===- hexagon_benchmark.h ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

/* Header file used to time a pipeline.
 *  This is meant to be used in device-standalone mode. */

#ifndef HEXAGON_BENCHMARK_H
#define HEXAGON_BENCHMARK_H
#include "HAP_perf.h"
#include "hexagon_types.h"

// Returns the average time, in microseconds, taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_time_us(int iterations, F op) {
  uint64_t start_time = HAP_perf_get_time_us();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_time = HAP_perf_get_time_us();
  return (uint64_t)((end_time - start_time) / iterations);
}

// Returns the average cycles taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_pcycles(int iterations, F op) {
  uint64_t start_cycle = HAP_perf_get_pcycles();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_cycle = HAP_perf_get_pcycles();
  return (uint64_t)((end_cycle - start_cycle) / iterations);
}
#endif
