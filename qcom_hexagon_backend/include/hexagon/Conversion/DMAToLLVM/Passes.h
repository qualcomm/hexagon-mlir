//===- Passes.h - Convert DMA to LLVM ops -----------------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_DMATOLLVM_PASSES_H
#define HEXAGON_CONVERSION_DMATOLLVM_PASSES_H
#include "DMAToLLVM.h"

namespace mlir {
namespace hexagon {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/DMAToLLVM/Passes.h.inc"

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_DMATOLLVM_PASSES_H
