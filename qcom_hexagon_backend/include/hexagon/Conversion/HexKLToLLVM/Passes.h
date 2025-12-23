//===- Passes.h - Convert HexKL to LLVM Ops  ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_H

#include "HexKLToLLVM.h"

namespace mlir {
namespace hexkl {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/HexKLToLLVM/Passes.h.inc"

} // namespace hexkl
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_H
