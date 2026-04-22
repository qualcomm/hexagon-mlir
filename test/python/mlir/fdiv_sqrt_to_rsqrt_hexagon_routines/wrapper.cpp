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

// Declare the external MLIR function
extern "C" _Float16 fdiv_sqrt_pattern_scalar_f16(_Float16 arg0);
extern "C" float fdiv_sqrt_pattern_scalar_f32(float arg1);

// Helper to generate random float
float get_rand(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

int main() {
  const int NUM_TESTS = 100;
  // Use positive inputs > 0 to avoid sqrt(0) or sqrt(negative) issues
  const float min_val = 0.1f;
  const float max_val = 10.0f;

  std::vector<_Float16> inputs_f16(NUM_TESTS);
  std::vector<float> inputs_f32(NUM_TESTS);
  std::vector<_Float16> results_f16(NUM_TESTS);
  std::vector<float> results_f32(NUM_TESTS);

  // Initialize random inputs
  for (int i = 0; i < NUM_TESTS; ++i) {
    inputs_f16[i] = (_Float16)get_rand(min_val, max_val);
    inputs_f32[i] = get_rand(min_val, max_val);
  }

  uint64_t avg_cycles = benchmark_time_us(1, [&]() {
    for (int i = 0; i < NUM_TESTS; ++i) {
      _Float16 res = fdiv_sqrt_pattern_scalar_f16(inputs_f16[i]);
      results_f16[i] = res;
    }
  });

  Result result = Result::Pass;

  for (int i = 0; i < NUM_TESTS; ++i) {
    float val0 = (float)inputs_f16[i];
    float rsqrt0 = 1.0f / std::sqrt(val0);
    _Float16 rsqrt_f16 = (_Float16)rsqrt0;
    float diff = (float)results_f16[i] - (float)rsqrt_f16;

    if (std::abs(diff) > 0.01f) {
      FARF(
          ALWAYS,
          "Mismatch at index %d: inputs(%f) -> got %f, expected %f (diff %f)\n",
          i, (float)inputs_f16[i], (float)results_f16[i], (float)rsqrt_f16,
          diff);
      result = Result::Fail;
      break;
    }
  }

  avg_cycles = benchmark_time_us(1, [&]() {
    for (int i = 0; i < NUM_TESTS; ++i) {
      float res = fdiv_sqrt_pattern_scalar_f32(inputs_f32[i]);
      results_f32[i] = res;
    }
  });

  for (int i = 0; i < NUM_TESTS; ++i) {
    float rsqrt_f32 = 1.0f / std::sqrt(inputs_f32[i]);
    float diff = results_f32[i] - rsqrt_f32;

    if (std::abs(diff) > 0.01f) {
      FARF(
          ALWAYS,
          "Mismatch at index %d: inputs(%f) -> got %f, expected %f (diff %f)\n",
          i, inputs_f32[i], results_f32[i], rsqrt_f32, diff);
      result = Result::Fail;
      break;
    }
  }

  if (result == Result::Pass) {
    FARF(ALWAYS, "Verification PASSED\n");
  } else {
    FARF(ALWAYS, "Verification FAILED\n");
  }

  TestReport tr("fdiv_sqrt_pattern_scalar", avg_cycles, "microseconds", result);
  tr.print();

  return result;
}
