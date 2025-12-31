//===- prof_utils.cc ------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "prof_utils.h"
#include <cstdint>
#include <fstream>
#include <sstream>

// Handler will instrument 1000 unique ID
#define LWP_COUNTER_SIZE 1000 // Can adjust the number of instrumented loops

// The lwp_buffer can have at max total 200 records (including start, end)
// associated with each ID. Example: A sibling loop is invoked 100 times-- each
// invocation has 2 hooks-- 200 records in total. Each record takes 3 words
// space. So total size will be at most LWP_COUNTER_SIZE * 200 * 3.
#define LWP_BUFFER_SIZE (LWP_COUNTER_SIZE * 200 * 3)

// Global profiling data

// The ID is used to index into the buffer.
// This buffer will track how many times an individual region is called.
__attribute__((visibility("hidden")))
uint32_t lwp_counter[LWP_COUNTER_SIZE] = {0};
// Instrumented data will be stored in this buffer.
__attribute__((visibility("hidden"))) uint32_t lwp_buffer[LWP_BUFFER_SIZE];
__attribute__((visibility("hidden"))) uint32_t lwp_buffer_size =
    LWP_BUFFER_SIZE;
__attribute__((visibility("hidden"))) uint32_t lwp_buffer_count = 0;

// Write profiling data to a JSON file
extern "C" bool WriteLWPOutput(const std::string &out_json) {
  std::ostringstream s;
  s << "{\n  \"entries\": [\n";

  for (size_t i = 0; i < lwp_buffer_count; i += 3) {
    uint64_t cycles =
        (static_cast<uint64_t>(lwp_buffer[i + 2]) << 32) | lwp_buffer[i + 1];
    s << "    {\n"
      << "      \"id\": " << lwp_buffer[i] << ",\n"
      << "      \"cyc\": " << cycles << "\n"
      << "    }";
    if (i + 3 < lwp_buffer_count)
      s << ",";
    s << "\n";
  }

  s << "  ]\n}\n";

  std::ofstream out(out_json);
  if (!out.is_open())
    return false;
  out << s.str();
  return out.good();
}
