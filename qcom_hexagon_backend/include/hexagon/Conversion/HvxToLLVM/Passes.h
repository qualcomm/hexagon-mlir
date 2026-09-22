//===- Passes.h - Convert Hvx to LLVM Ops ----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HVXTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_HVXTOLLVM_PASSES_H

#include "HvxToLLVM.h"

namespace mlir {
namespace hvx {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/HvxToLLVM/Passes.h.inc"

} // namespace hvx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HVXTOLLVM_PASSES_H
