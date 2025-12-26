//===- Passes.h - Linalg lowering and optimization passes -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_TRANSFORMS_PASSES_H
#define HEXAGON_TRANSFORMS_PASSES_H

#include "Transforms.h"

namespace mlir {
namespace hexagon {

#define GEN_PASS_REGISTRATION
#include "hexagon/Transforms/Passes.h.inc"

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_TRANSFORMS_PASSES_H
