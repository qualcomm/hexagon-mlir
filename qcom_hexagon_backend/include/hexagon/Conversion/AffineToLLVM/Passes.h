//===- Passes.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_AFFINE_TO_LLVM_CONVERSION_PASSES_H
#define HEXAGON_AFFINE_TO_LLVM_CONVERSION_PASSES_H

#include "AffineToLLVM.h"

namespace mlir {
namespace hexagon {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_AFFINE_TO_LLVM_CONVERSION_PASSES_H
