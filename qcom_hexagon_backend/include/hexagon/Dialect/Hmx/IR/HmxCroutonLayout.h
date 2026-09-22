//===-- HmxCroutonLayout.h - the crouton layout's numbers -------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The one home for the crouton's physical shape. `#hmx.crouton`'s parameters
// (HmxAttrs.td) are aliases of these, the passes and the runtime-leaf lowering
// read them from here, and the host prepack contract is built from them -- so a
// change to the engine's tile can only be made in one place.
//
// The layout contract itself (why one crouton is 16 pairs x 32 columns x 2
// halves, and why AH/WH/AR are the same permutation) is documented with the
// dialect: include/hexagon/Dialect/Hmx/IR/HmxDialect.h and HmxAttrs.td.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMXCROUTONLAYOUT_H
#define HEXAGON_DIALECT_HMX_IR_HMXCROUTONLAYOUT_H

#include <cstdint>

namespace mlir {
namespace hmx {
namespace crouton {

/// The edge of one crouton: a 32x32 fp16 tile. Every extent the engine touches
/// is a multiple of it.
constexpr int64_t kTileEdge = 32;

/// The tail of one crouton in the physical rank-5 type: 16 pairs x 32 columns x
/// 2 halves of a pair.
constexpr int64_t kCroutonPair = 16;
constexpr int64_t kCroutonCol = 32;
constexpr int64_t kCroutonHalf = 2;

/// One crouton as an element count and as a byte count (f16).
constexpr int64_t kCroutonElements = kCroutonPair * kCroutonCol * kCroutonHalf;
constexpr int64_t kCroutonBytes = kCroutonElements * 2;

} // namespace crouton
} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_IR_HMXCROUTONLAYOUT_H
