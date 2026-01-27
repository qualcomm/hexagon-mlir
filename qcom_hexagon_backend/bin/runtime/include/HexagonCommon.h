//===- HexagonCommon.h - Common helper print functions --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCOMMON_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCOMMON_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

#if defined(__hexagon__)
#include <HAP_farf.h>
#define HEXAGON_PRINT(level, ...) FARF(level, __VA_ARGS__)
#else
#include <cstdio>
#define HEXAGON_PRINT(level, ...) printf(__VA_ARGS__)
#endif

#define CHUNK_SIZE 2048

#define HEXAGON_SAFE_CALL(api_call)                                            \
  do {                                                                         \
    int result = api_call;                                                     \
    if (result != 0) {                                                         \
      HEXAGON_PRINT(ERROR, "ERROR: " #api_call " failed with error %d.",       \
                    result);                                                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

constexpr int kHexagonAllocAlignment = 2048;

inline void CheckFailed(const char *expr, const char *file, int line,
                        const std::string &msg = "") {
  std::cerr << "Error: " << expr << " failed at " << file << ":" << line;
  if (!msg.empty()) {
    std::cerr << " - " << msg;
  }
  std::cerr << std::endl;
  abort();
}

#ifndef CHECK
#define CHECK(x, ...)                                                          \
  if (!(x)) {                                                                  \
    std::ostringstream oss;                                                    \
    oss << #__VA_ARGS__;                                                       \
    CheckFailed(#x, __FILE__, __LINE__, oss.str());                            \
  }
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(ret, val)                                                     \
  if ((ret) != (val)) {                                                        \
    CheckFailed(#ret " != " #val, __FILE__, __LINE__);                         \
  }
#endif

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCOMMON_H_
