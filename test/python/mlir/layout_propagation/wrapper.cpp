//===-- wrapper.cpp -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

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

#include "CRunnerUtils.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

extern "C" void unpack_then_reduce(int *, int *, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int *, int *, int64_t, int64_t,
                                   int64_t);

extern "C" void reduce_then_unpack(int *, int *, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int *, int *, int64_t, int64_t,
                                   int64_t);

template <typename T>
int check(T *out_reduce_unpack, T *out_unpack_reduce, const uint32_t dim_out) {
  int errcnt = 0;
  printf("Checking...\n");
  for (int i = 0; i < dim_out; i++) {
    printf("Elem %i: out_reduce_unpack=%i, out_unpack_reduce=%i\n", i,
           out_reduce_unpack[i], out_unpack_reduce[i]);
    if (out_reduce_unpack[i] != out_unpack_reduce[i]) {
      printf("Missmatch elem %i: out_reduce_unpack=%i, out_unpack_reduce=%i\n",
             i, out_reduce_unpack[i], out_unpack_reduce[i]);
      errcnt += 1;
    }
  }
  return errcnt;
}

template <typename T> void set_rand(T *ptr, uint32_t len) {
  const int range_min = 1;
  const int range_max = 100;

  std::mt19937 mt_eng(0);
  std::uniform_int_distribution<int> dist(range_min, range_max);

  for (size_t i = 0; i < len; i++) {
    ptr[i] = dist(mt_eng);
  }
}
template <typename T> void set_zero(T *ptr, uint32_t len) {
  for (size_t i = 0; i < len; i++)
    ptr[i] = 0;
}

int main(int argc, char **argv) {

  std::string bname = "layout_propagation";

  const uint32_t dim0_in = 2;
  const uint32_t dim1_in = 2;
  const uint32_t dim2_in = 4;
  const uint32_t dim3_in = 4;
  const uint32_t dim0_out = 8;

  Tensor<int, 4> in(
      {{dim0_in, dim1_in, dim2_in, dim3_in},
       {dim1_in * dim2_in * dim3_in, dim2_in * dim3_in, dim3_in, 1},
       128},
      MemType::HEAP);
  Tensor<int, 1> out_unpack_reduce({{dim0_out}, {1}, 128}, MemType::HEAP);
  Tensor<int, 1> out_reduce_unpack({{dim0_out}, {1}, 128}, MemType::HEAP);

  MemRefDescriptor<int, 4> *dt_in = in.toMemRefDesc();
  MemRefDescriptor<int, 1> *dt_out_unpack_reduce =
      out_unpack_reduce.toMemRefDesc();
  MemRefDescriptor<int, 1> *dt_out_reduce_unpack =
      out_reduce_unpack.toMemRefDesc();

  const uint32_t num_elems = dim0_in * dim1_in * dim2_in * dim3_in;
  set_rand(dt_in->allocated, num_elems);
  set_zero(dt_out_unpack_reduce->allocated, dim0_out);
  set_zero(dt_out_reduce_unpack->allocated, dim0_out);

  uint64_t avg_cycles_unpack_reduce = benchmark_time_us(50, [&]() {
    unpack_then_reduce(dt_in->allocated, dt_in->allocated, 0, dim0_in, dim1_in,
                       dim2_in, dim3_in, dim1_in * dim2_in * dim3_in,
                       dim2_in * dim3_in, dim3_in, 1,
                       dt_out_unpack_reduce->allocated,
                       dt_out_unpack_reduce->allocated, 0, dim0_out, 1);
  });

  uint64_t avg_cycles_reduce_unpack = benchmark_time_us(50, [&]() {
    reduce_then_unpack(dt_in->allocated, dt_in->allocated, 0, dim0_in, dim1_in,
                       dim2_in, dim3_in, dim1_in * dim2_in * dim3_in,
                       dim2_in * dim3_in, dim3_in, 1,
                       dt_out_reduce_unpack->allocated,
                       dt_out_reduce_unpack->allocated, 0, dim0_out, 1);
  });

  printf("Time unpack->reduce: %" PRIu64 "\n", avg_cycles_unpack_reduce);
  printf("Time reduce->unpack: %" PRIu64 "\n", avg_cycles_reduce_unpack);

  Result result = Result::Pass;
  int mismatches = check(dt_out_reduce_unpack->allocated,
                         dt_out_unpack_reduce->allocated, dim0_out);
  if (mismatches) {
    result = Result::Fail;
    printf("Failed layout prop test\n");
  }

  TestReport tr(bname, avg_cycles_unpack_reduce + avg_cycles_reduce_unpack,
                "microseconds", result);
  tr.save();
  return 0;
}
