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

extern "C" void hexkl_attention(int, int64_t, MemRefDescriptor<_Float16, 4> *,
                                int64_t, MemRefDescriptor<_Float16, 4> *,
                                int64_t, MemRefDescriptor<_Float16, 4> *,
                                int64_t, MemRefDescriptor<float, 4> *, int, int,
                                int, int, int, int);

#define MAX_DIFF 1e-3

constexpr int Z = 1;
constexpr int H = 1;
constexpr int N_CTX = 1024;
constexpr int D_HEAD = 64;
constexpr int NUM_ELEMENTS = Z * H * N_CTX * D_HEAD;

// Sets Query, Key, and Value matrices which are of dimension (Z, H, N_CTX,
// D_HEAD) ~ N(0, 0.5)
void set_pointer_random(_Float16 *ptr) {
  const float std = 0.5f;
  const float mean = 0.0f;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  for (size_t i = 0; i < NUM_ELEMENTS; i++)
    ptr[i] = _Float16(roundf(dist(gen) * 10000) /
                      10000); // round off to 4 decimal places
}

// Expects an input of (Z, H, N_CTX, D_HEAD)
// Returns input.transpose(0, 1, 3, 2)
// Output dims: (Z, H, D_HEAD, N_CTX)
void transpose(_Float16 *input, _Float16 *output) {
  for (size_t z = 0; z < Z; ++z) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t n = 0; n < N_CTX; ++n) {
        for (size_t d = 0; d < D_HEAD; ++d) {
          size_t input_idx = (z * H * N_CTX * D_HEAD) + (h * N_CTX * D_HEAD) +
                             (n * D_HEAD) + d;
          size_t output_idx =
              (z * H * N_CTX * D_HEAD) + (h * N_CTX * D_HEAD) + (d * N_CTX) + n;
          _Float16 _Float16_input = (input[input_idx]);
          output[output_idx] = _Float16_input;
        }
      }
    }
  }
}

// input_dim_2 and input_dim_3 are dimensions 2 and 3 of input matrix A
// output_dim_2 and output_dim_3 are dimensions 2 and 3 of expected output
// matrix Assumes the first two dimensions of all matrices are Z and H
// respectively
void matMul(_Float16 *A, _Float16 *B, float *output, int input_dim_2,
            int input_dim_3, int output_dim_2, int output_dim_3) {
  for (size_t z = 0; z < Z; ++z) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t i = 0; i < output_dim_2; ++i) {
        for (size_t j = 0; j < output_dim_3; ++j) {
          // reduce along input_dim_3
          float acc = 0.0f;
          for (size_t k = 0; k < input_dim_3; ++k) {
            _Float16 op1 =
                (A[(z * H * input_dim_2 * input_dim_3) +
                   (h * input_dim_2 * input_dim_3) + (i * input_dim_3) + k]);
            _Float16 op2 =
                (B[(z * H * input_dim_3 * output_dim_3) +
                   (h * input_dim_3 * output_dim_3) + (k * output_dim_3) + j]);
            acc += op1 * op2;
          }
          output[(z * H * output_dim_2 * output_dim_3) +
                 (h * output_dim_2 * output_dim_3) + (i * output_dim_3) + j] =
              (float)acc;
        }
      }
    }
  }
}

// Assumes dimensions of input matrix is (Z, H, N_CTX, N_CTX)
// Multiplies input elemwise by 0.5
void multiplySoftmaxInputByScalar(float *input, float scale) {
  for (size_t z = 0; z < Z; ++z) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t n = 0; n < N_CTX; ++n) {
        for (size_t d = 0; d < N_CTX; ++d) {
          size_t idx =
              (z * H * N_CTX * N_CTX) + (h * N_CTX * N_CTX) + (n * N_CTX) + d;
          float tmp_val = (input[idx]);
          input[idx] = tmp_val * scale;
        }
      }
    }
  }
}

// Assumes dimensions of input matrix is (Z, H, N_CTX, N_CTX)
// Computes Softmax about last axis
void computeSoftmax(float *input, float *output) {
  for (size_t z = 0; z < Z; ++z) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t n = 0; n < N_CTX; ++n) {
        float max_val = std::numeric_limits<float>::min();
        for (size_t d = 0; d < N_CTX; ++d) {
          float tmp_input = input[(z * H * N_CTX * N_CTX) +
                                  (h * N_CTX * N_CTX) + (n * N_CTX) + d];
          max_val = std::max(max_val, tmp_input);
        }
        float exp_sum = 0.0;
        for (size_t d = 0; d < N_CTX; ++d) {
          float tmp_input = (input[(z * H * N_CTX * N_CTX) +
                                   (h * N_CTX * N_CTX) + (n * N_CTX) + d]);
          exp_sum += std::exp(tmp_input - max_val);
        }
        for (size_t d = 0; d < N_CTX; ++d) {
          float tmp_input = (input[(z * H * N_CTX * N_CTX) +
                                   (h * N_CTX * N_CTX) + (n * N_CTX) + d]);
          output[(z * H * N_CTX * N_CTX) + (h * N_CTX * N_CTX) + (n * N_CTX) +
                 d] = std::exp(tmp_input - max_val) / exp_sum;
        }
      }
    }
  }
}

// Reference Implementation matches the input MLIR for flash attention.
// Only the matmul uses fp16 inputs and f32 outputs and rest of the operations
// are on f32 operands.
bool check(_Float16 *Query, _Float16 *Key, _Float16 *Value, float *Output) {
  bool ret = true;
  int numSoftmaxInputElems = Z * H * N_CTX * N_CTX;
  _Float16 *keyTranspose =
      (_Float16 *)memalign(128, NUM_ELEMENTS * sizeof(_Float16));
  float *softmaxInput =
      (float *)memalign(128, numSoftmaxInputElems * sizeof(float));
  float *softmaxOutput =
      (float *)memalign(128, numSoftmaxInputElems * sizeof(float));
  float *finalOutput = (float *)memalign(128, NUM_ELEMENTS * sizeof(float));

  transpose(Key, keyTranspose); // K -> KT
  matMul(Query, keyTranspose, softmaxInput, N_CTX, D_HEAD, N_CTX,
         N_CTX); // S = MatMul (Q, KT)
  multiplySoftmaxInputByScalar(softmaxInput,
                               0.5); // // Multiply softmax input by 0.5
  computeSoftmax(softmaxInput, softmaxOutput); // P = Softmax(S)
  matMul((_Float16 *)softmaxOutput, Value, finalOutput, N_CTX, N_CTX, N_CTX,
         D_HEAD); // MatMul (P, V)

  for (size_t i = 0; i < NUM_ELEMENTS; i++) {
    float tritonOut = (Output[i]);
    float cpuOut = (finalOutput[i]);
    float diff = tritonOut - cpuOut;
    if (diff > MAX_DIFF || diff < -MAX_DIFF) {
      FARF(ALWAYS,
           "Mismatch at index %d, CPU output was %f and Triton out was %f\n", i,
           float(cpuOut), float(tritonOut));
      ret = false;
      break;
    }
  }
  free(keyTranspose);
  free(softmaxInput);
  free(softmaxOutput);
  free(finalOutput);
  return ret;
}

int main(int argc, char **argv) {
  const MemType memType = MemType::HEAP;

  int S0 = 0;
  Tensor<_Float16, 4> Q({{Z, H, N_CTX, D_HEAD},
                         {H * N_CTX * D_HEAD, N_CTX * D_HEAD, D_HEAD, 1},
                         128},
                        memType); // Query
  Tensor<_Float16, 4> K({{Z, H, N_CTX, D_HEAD},
                         {H * N_CTX * D_HEAD, N_CTX * D_HEAD, D_HEAD, 1},
                         128},
                        memType); // Key
  Tensor<_Float16, 4> V({{Z, H, N_CTX, D_HEAD},
                         {H * N_CTX * D_HEAD, N_CTX * D_HEAD, D_HEAD, 1},
                         128},
                        memType); // Value
  Tensor<float, 4> Output({{Z, H, N_CTX, D_HEAD},
                           {H * N_CTX * D_HEAD, N_CTX * D_HEAD, D_HEAD, 1},
                           128},
                          memType); // Output

  Ty_MemRefDescHalf4D *dQ = Q.toMemRefDesc();
  Ty_MemRefDescHalf4D *dK = K.toMemRefDesc();
  Ty_MemRefDescHalf4D *dV = V.toMemRefDesc();
  Ty_MemRefDescFloat4D *dOutput = Output.toMemRefDesc();

  set_pointer_random(dQ->allocated);
  set_pointer_random(dK->allocated);
  set_pointer_random(dV->allocated);

  uint64_t avg_time_us = benchmark_time_us(1, [&]() {
    hexkl_attention(S0, 4, dQ, 4, dK, 4, dV, 4, dOutput, 1, 1, 1, 0, 0, 0);
  });

  Result result = Result::Pass;
  if (!check(dQ->allocated, dK->allocated, dV->allocated, dOutput->allocated))
    result = Result::Fail;
  TestReport tr("hexkl_attention", avg_time_us, "microseconds", result);
  tr.save();
  return result;
}
