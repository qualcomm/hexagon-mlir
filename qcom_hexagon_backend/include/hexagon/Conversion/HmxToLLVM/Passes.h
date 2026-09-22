//===- Passes.h - Convert Hmx to LLVM Ops ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HMXTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_HMXTOLLVM_PASSES_H

#include "HmxToLLVM.h"

namespace mlir {
namespace hmx {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/HmxToLLVM/Passes.h.inc"

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HMXTOLLVM_PASSES_H
