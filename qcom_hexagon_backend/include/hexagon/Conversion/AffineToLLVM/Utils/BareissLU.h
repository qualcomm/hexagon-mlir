//===- BareissLU.h -                       --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_BAREISSLU_H
#define HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_BAREISSLU_H

#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {
namespace hexagon {

/**
 * Bareiss LU decomposition, for linear matroids.
 * Supports updates (erase/insert) followed by a rebuild call.
 * Once the basis is built, independence/circuit queries are solved in O(m r +
 * r^2) time.
 *
 * Decomposition of A (whose columns are a subset of source^T) satisfies:
 * P A = L D^-1 U.
 */
struct BareissLU {
  // Base matrix (n x m). Matroid elements are rows of A.
  mlir::presburger::IntMatrix source;
  // Independent set.
  llvm::SmallVector<unsigned> I;
  // Row permutation. (length m)
  llvm::SmallVector<unsigned> P;
  // Lower matrix. (m x r)
  mlir::presburger::IntMatrix L;
  // Upper matrix; transformed basis. (r x n)
  mlir::presburger::IntMatrix U;
  // Denominators.
  llvm::SmallVector<llvm::DynamicAPInt> D;
  // Pivot columns.
  llvm::SmallVector<unsigned> pivotCols;

  /**
   * Initializes with an empty set of rows, using the given source matrix.
   * @param source the matrix from which all rows are taken.
   */
  BareissLU(mlir::presburger::IntMatrix &&source);
  /**
   * Updates I with the symmetric difference of rows, and computes the LU
   * decomposition.
   * @param rows the rows to remove and insert. Must be sorted.
   * @return the rank of the new matrix.
   */
  unsigned update(llvm::ArrayRef<unsigned> rows);
  /**
   * Checks dependence with a given row of source.
   * Returns true if dependent, and sets out to the index of rows in source
   * on which it is dependent. i.e., it solves
   * A x = source[j],
   * and returns { I[i] | x[i] != 0 }.
   * @param j the index of the row in source to check.
   * @param[out] out will be filled with the indices of source rows on which row
   * j is dependent.
   */
  bool checkDependence(unsigned j, llvm::SmallVectorImpl<unsigned> *out);
};

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_AFFINETOLLVM_UTILS_BAREISSLU_H
