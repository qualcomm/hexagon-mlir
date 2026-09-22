//===- HmxDialect.cpp - HMX Dialect  --------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::hmx;
/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom attrs, types and operations for the dialect.
void HmxDialect::initialize() {
  registerOperations();
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/IR/HmxDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Layout contract (see include/hexagon/Dialect/Hmx/IR/HmxDialect.h)
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hmx {

// One layout, three roles: AH/WH/AR are uses, not types. The layout is named by
// the `#hmx.crouton` encoding on the type (rather than inferred from the rank-5
// shape), so a rank-5 shape without the encoding is *not* a crouton -- that is
// what keeps a stray shape from being mistaken for the layout.

// The weight grid is the activation grid transposed: the engine walks the grid's
// second dim in every role, so a `[K, N]` weight is stored as its `[N, K]`
// transpose with Wᵀ's K on dim1. `croutonLayoutType` is therefore fed the
// transposed logical shape; the encoding is the ordinary `#hmx.crouton` (the
// role is a property of the use, not of the type). See
// docs/hmx/hmx-weight-layout-plan.md section 0.
RankedTensorType weightCroutonType(RankedTensorType weightRowMajorKN) {
  assert(weightRowMajorKN.getRank() == 2 &&
         "weight crouton layout needs a rank-2 [K, N] weight");
  int64_t k = weightRowMajorKN.getDimSize(0);
  int64_t n = weightRowMajorKN.getDimSize(1);
  return croutonLayoutType(
      RankedTensorType::get({n, k}, weightRowMajorKN.getElementType()));
}

int64_t weightKTiles(ShapedType weightCrouton) {
  return weightCrouton.getDimSize(1);
}

int64_t weightNTiles(ShapedType weightCrouton) {
  return weightCrouton.getDimSize(0);
}

} // namespace hmx
} // namespace mlir
