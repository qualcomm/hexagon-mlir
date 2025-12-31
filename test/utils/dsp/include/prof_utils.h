//===- prof_utils.h -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef LWP_PROFILER_H
#define LWP_PROFILER_H

#include <cstdint>
#include <string>

extern "C" bool WriteLWPOutput(const std::string &out_json);

#endif // LWP_PROFILER_H
