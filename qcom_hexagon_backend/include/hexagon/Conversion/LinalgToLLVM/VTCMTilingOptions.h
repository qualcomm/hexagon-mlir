//===- VTCMTilingOptions.h - tiling for vtcm options ----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_VTCMTILINGOPTIONS_H
#define HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_VTCMTILINGOPTIONS_H
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
namespace hexagon {

// Given a linalg op and (optionally) user defined tile sizes, returns tiling
// options (if valid) for the vtcm constraints.
FailureOr<linalg::LinalgTilingOptions>
getVTCMTilingOptions(linalg::LinalgOp op,
                     std::optional<SmallVector<int64_t>> userProvidedTileSizes,
                     SmallVector<bool> &fullTensorOperands);

SmallVector<int64_t> getInitialTileSize(linalg::LinalgOp);
std::optional<SmallVector<int64_t>>
determineTileSizes(linalg::LinalgOp, SmallVector<int64_t> tilingDims = {});
} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_VTCMTILINGOPTIONS_H
