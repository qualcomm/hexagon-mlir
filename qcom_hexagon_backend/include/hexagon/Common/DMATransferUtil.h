//===- DMATransferUtil.h - some useful general DMA functions --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_COMMON_DMA_TRANSFER_UTIL_H
#define HEXAGON_COMMON_DMA_TRANSFER_UTIL_H

namespace mlir {
namespace hexagon {

// Given a `view` that is a ranked memref type of Rank <= 2,
// computes and returns number of elements in the memref.
// TODO : For Rank > 2 we need to refactor DMA runtime first.
Value createNumElements(Location loc, IRRewriter &rewriter, Value view);

// Create DMA start op to initiate asynchronous data transfer
// from `source` to `target` memref using the specified tag.
// Returns true if all checks pass and dma_start is created.
bool createDMAStartOp(Location loc, IRRewriter &rewriter, Value source,
                      Value target, Value tag);

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_COMMON_DMA_TRANSFER_UTIL_H