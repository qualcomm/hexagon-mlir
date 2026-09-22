//===- Passes.h - HMX dialect transform pass registration -------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_H

#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

namespace mlir {
namespace hmx {

#define GEN_PASS_REGISTRATION
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_H
