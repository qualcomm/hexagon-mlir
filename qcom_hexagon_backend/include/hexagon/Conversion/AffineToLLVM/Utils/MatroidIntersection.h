//===- MatroidIntersection.h - matroid intersection         ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_MATROIDINTERSECTION_H
#define HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_MATROIDINTERSECTION_H

#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {
namespace hexagon {

/**
 * Computes the maximum independent set of a linear and partition matroid.
 * If prefixOnly is true, the result must have that the groups form a prefix.
 * Runtime is O(n^2 r + n r^3).
 * @param mat matrix, the rows of which the linear matroid is formed.
 * mat.getNumRows() must equal groups.size().
 * @param groups groups for the partition matroid. A set I is independent if
 * groups[i] are all unique for i in I. Should be sorted.
 * @param prefixOnly if true, only independent sets that are a prefix of the
 * groups are considered.
 * @return the row indices that form the maximum independent set (prefix)
 */
llvm::SmallVector<unsigned>
matroidIntersectionLP(mlir::presburger::IntMatrix &&mat,
                      llvm::ArrayRef<unsigned> groups, bool prefixOnly = true);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_MATROIDINTERSECTION_H
