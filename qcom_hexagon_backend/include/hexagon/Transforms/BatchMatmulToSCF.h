//===- BatchMatmulToSCF.h - Pattern to map batch matmul to HMX ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_BATCH_MATMUL_TO_SCF_H
#define HEXAGON_TRANSFORMS_BATCH_MATMUL_TO_SCF_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {

/// Rewrite pattern to lower linalg.batch_matmul to scf.for { linalg.matmul } if
/// the operations will lower to HMX
struct LoopedHMXMatmul final : public OpRewritePattern<linalg::BatchMatmulOp> {

  LoopedHMXMatmul(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  /// Create a linalg extract slice operation
  TypedValue<RankedTensorType>
  extractSlice(PatternRewriter &rewriter, Location loc,
               TypedValue<IndexType> select,
               TypedValue<RankedTensorType> tensor) const;

  /// Create a linalg insert slice operation
  TypedValue<RankedTensorType>
  insertSlice(PatternRewriter &rewriter, Location loc,
              TypedValue<IndexType> select, TypedValue<RankedTensorType> source,
              TypedValue<RankedTensorType> dest) const;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const;
};

} // namespace mlir

#endif
