//===- SplitReductionUtils.h - utility functions for split reduction ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_SPLITREDUCTIONUTILS_H
#define HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_SPLITREDUCTIONUTILS_H
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace hexagon {

// Given a reduction loop, apply split reduction to convert it into a
// parallel-reduction loop, thereby creating more opportunities for
// vectorization.
LogicalResult SplitReductionLinalgOp(linalg::LinalgOp generalizeOp);
bool IsSplitReductionCandidate(linalg::LinalgOp op);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_SPLITREDUCTIONUTILS_H
