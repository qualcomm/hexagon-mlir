//===- ThreadDistribution.h - Thread-level parallelism utilities ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for distributing work across virtual threads. Used by
// FormVirtualThreadsPass (linalg.generic) and
// FormCompositeVirtualThreadsPass (composite ops).
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_LINALGTOLLVM_THREADDISTRIBUTION_H
#define HEXAGON_CONVERSION_LINALGTOLLVM_THREADDISTRIBUTION_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace hexagon {

/// Minimum iteration-space size below which multi-threading overhead
/// is not worth the benefit.
inline constexpr int64_t kMinPolytopeSize = 65536;

/// Given iterator types, static loop ranges, and thread configuration,
/// compute SCF tiling options for thread-level parallelism.
///
/// The caller extracts iterator types and ranges from whatever op type it
/// has (linalg::GenericOp, TilingInterface, etc.) and passes them as plain
/// arrays. This function contains only the thread-selection logic.
///
/// \param rewriter       The IR rewriter (needed for OpFoldResult creation).
/// \param iteratorTypes  Iterator types (parallel / reduction) per loop dim.
/// \param staticRanges   Static loop ranges per dim (no dynamic dims allowed).
/// \param polytopeSize   Pre-computed work size for the "is it worth it"
///                       heuristic (>= kMinPolytopeSize to proceed).
/// \param blockSize      User-override tile size for testing (0 = auto).
/// \param numThreads     Preferred number of threads.
/// \param tilingOptions  [out] Populated with tile sizes, interchange, and
///                       ForallOp loop type if threading is beneficial.
/// \returns true if tiling options were set, false if threading should be
///          skipped for this op.
bool computeTLPTilingOptions(IRRewriter &rewriter,
                             ArrayRef<utils::IteratorType> iteratorTypes,
                             ArrayRef<int64_t> staticRanges,
                             int64_t polytopeSize, int blockSize,
                             int numThreads,
                             scf::SCFTilingOptions &tilingOptions);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_LINALGTOLLVM_THREADDISTRIBUTION_H
