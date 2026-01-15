//===- HexagonCustomDefs.cpp - --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <cstdlib> // For std::malloc and std::free
#include <cstring>

namespace llvm {
int DisableABIBreakingChecks = 0; // Define the variable
int EnableABIBreakingChecks = 0;  // Define the variable

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

uint64_t xxh3_64bits(ArrayRef<uint8_t> data) {
  const uint8_t *input = data.data();
  size_t length = data.size();
  uint64_t hash = length;

  if (length >= 32) {
    const uint8_t *end = input + length;
    const uint8_t *limit = end - 32;
    uint64_t v1 = PRIME64_1 + PRIME64_2;
    uint64_t v2 = PRIME64_2;
    uint64_t v3 = 0;
    uint64_t v4 = -PRIME64_1;

    do {
      v1 += *reinterpret_cast<const uint64_t *>(input) * PRIME64_2;
      v1 = (v1 << 31) | (v1 >> (64 - 31));
      v1 *= PRIME64_1;
      input += 8;

      v2 += *reinterpret_cast<const uint64_t *>(input) * PRIME64_2;
      v2 = (v2 << 31) | (v2 >> (64 - 31));
      v2 *= PRIME64_1;
      input += 8;

      v3 += *reinterpret_cast<const uint64_t *>(input) * PRIME64_2;
      v3 = (v3 << 31) | (v3 >> (64 - 31));
      v3 *= PRIME64_1;
      input += 8;

      v4 += *reinterpret_cast<const uint64_t *>(input) * PRIME64_2;
      v4 = (v4 << 31) | (v4 >> (64 - 31));
      v4 *= PRIME64_1;
      input += 8;
    } while (input <= limit);

    hash = (v1 << 1) | (v1 >> (64 - 1));
    hash += (v2 << 7) | (v2 >> (64 - 7));
    hash += (v3 << 12) | (v3 >> (64 - 12));
    hash += (v4 << 18) | (v4 >> (64 - 18));
  }

  hash += PRIME64_5;
  hash ^= hash >> 33;
  hash *= PRIME64_2;
  hash ^= hash >> 29;
  hash *= PRIME64_3;
  hash ^= hash >> 32;

  return hash;
}
[[noreturn]] void report_bad_alloc_error(const char *Reason,
                                         bool /*GenCrashDiag*/) {
  // Directly write an out-of-memory message to a fixed memory location and
  // abort. const char *OOMMessage = "LLVM ERROR: out of memory\n"; const char
  // *Newline = "\n";

  // Assuming you have a way to output or log messages in your system, replace
  // this part with your logging mechanism. For example, writing to a specific
  // memory-mapped I/O location or using a custom logging function. Here, we
  // just simulate the process by assigning to volatile pointers. volatile char
  // *output = const_cast<volatile char*>(OOMMessage); output =
  // const_cast<volatile char*>(Reason); output = const_cast<volatile
  // char*>(Newline);

  std::abort();
}

} // namespace llvm

extern "C" __attribute__((visibility("default"))) void *
aligned_alloc(size_t alignment, size_t size) {
  return memalign(alignment, size);
}
