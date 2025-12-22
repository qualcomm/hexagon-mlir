//===- Passes.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_H

#include "HexagonMemToLLVM.h"

namespace mlir {
namespace hexagonmem {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h.inc"

} // namespace hexagonmem
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_H
