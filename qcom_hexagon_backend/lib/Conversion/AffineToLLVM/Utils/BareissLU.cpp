//===------------ BareissLU.cpp - Bareiss LU decomposition ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a fraction-free LU decomposition of integer matrices.
// It is designed to be used in matroid intersections, so it supports updates
// and independence and circuit queries.
//===----------------------------------------------------------------------===//

/*
 * The algorithm used is based on Middeke et al. Common Factors in Fraction-Free
 * Matrix Decompositions (2021) Algorithm 4, slightly modified to avoid swapping
 * columns and to handle rank-deficient input matrices. A circuit-finding oracle
 * simply reduces to finding the support in the row space.
 */

#include "hexagon/Conversion/AffineToLLVM/Utils/BareissLU.h"

using namespace mlir;
using namespace mlir::presburger;
using namespace mlir::hexagon;
using namespace llvm;

BareissLU::BareissLU(IntMatrix &&source)
    : source(std::forward<IntMatrix>(source)),
      L(this->source.getNumColumns(), 0), U(0, 0) {}

unsigned BareissLU::update(llvm::ArrayRef<unsigned> rows) {
  // First, update I with symmetric difference.
  SmallVector<unsigned> temp;
  temp.reserve(I.size() + rows.size());
  {
    unsigned i = 0, j = 0;
    while (i < I.size() && j < rows.size()) {
      unsigned x = I[i], y = rows[j];
      if (x < y) {
        temp.push_back(x);
        i++;
      } else if (x > y) {
        temp.push_back(y);
        j++;
      } else {
        i++;
        j++;
      }
    }
    while (i < I.size()) {
      temp.push_back(I[i]);
      i++;
    }
    while (j < rows.size()) {
      temp.push_back(rows[j]);
      j++;
    }
  }
  I = std::move(temp);
  // Construct matrix A.
  unsigned m = source.getNumColumns();
  unsigned n = I.size();
  L.resize(m, m);
  for (unsigned i = 0; i < m; i++) {
    L.fillRow(i, 0);
  }
  U.resize(m, n);
  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = 0; j < m; j++) {
      U.at(j, i) = source.at(I[i], j);
    }
  }
  P.resize(m);
  std::iota(P.begin(), P.end(), 0);
  D.clear();
  pivotCols.clear();
  unsigned r = 0;
  DynamicAPInt prev(1); // Previous pivot.
  // Main LU loop.
  for (unsigned k = 0; k < n; k++) {
    unsigned pivotRow = r;
    for (; pivotRow < m; pivotRow++) {
      if (U.at(pivotRow, k) != 0) {
        break;
      }
    }
    if (pivotRow == m) {
      // No pivot; skip column.
      // Shouldn't happen if columns are independent, though.
      continue;
    }
    if (pivotRow != r) {
      U.swapRows(r, pivotRow);
      L.swapRows(r, pivotRow);
      std::swap(P[r], P[pivotRow]);
    }
    DynamicAPInt pivot = U.at(r, k);
    L.at(r, r) = pivot;
    for (unsigned i = r + 1; i < m; i++) {
      DynamicAPInt mult = U.at(i, k);
      L.at(i, r) = mult;
      U.at(i, k) = 0;
      for (unsigned j = k + 1; j < n; j++) {
        DynamicAPInt num = pivot * U.at(i, j) - mult * U.at(r, j);
        assert(num % prev == 0);
        U.at(i, j) = num / prev;
      }
    }
    D.push_back(prev * pivot);
    prev = pivot;
    r++;
    pivotCols.push_back(k);
  }
  L.removeColumns(r, m - r);
  U.removeRows(r, m - r);
  return r;
}

bool BareissLU::checkDependence(unsigned j,
                                llvm::SmallVectorImpl<unsigned> *out) {
  if (out) {
    out->clear();
  }
  if (I.size() == 0) {
    // Return true if all zeros.
    auto row = source.getRow(j);
    for (const auto &el : row) {
      if (el != 0) {
        return false;
      }
    }
    return true;
  }
  assert(source.getNumColumns() == P.size());
  llvm::SmallVector<DynamicAPInt> y(P.size());
  for (unsigned i = 0; i < P.size(); i++) {
    y[i] = source.at(j, P[i]);
  }
  unsigned r = D.size();
  unsigned m = L.getNumRows();
  unsigned n = U.getNumColumns();
  // Forward solve D^-1 L x = y.
  DynamicAPInt prev(1);
  for (unsigned k = 0; k < r; k++) {
    DynamicAPInt pivot = L.at(k, k);
    for (unsigned i = k + 1; i < m; i++) {
      DynamicAPInt num = pivot * y[i] - L.at(i, k) * y[k];
      assert(num % prev == 0);
      y[i] = num / prev;
    }
    prev = pivot;
  }
  for (unsigned i = r; i < m; i++) {
    if (y[i] != 0) {
      return false;
    }
  }
  if (!out) {
    // Can skip backward solve.
    return true;
  }
  // Backward solve.
  SmallVector<Fraction> x(n);
  for (unsigned i = r; i-- > 0;) {
    unsigned j = pivotCols[i];
    Fraction s = 0;
    for (unsigned k = j + 1; k < n; k++) {
      s += U.at(i, k) * x[k];
    }
    DynamicAPInt den = U.at(i, j);
    assert(den != 0);
    x[j] = (y[i] - s) / den;
  }
  // Set out to row indices.
  for (unsigned i = 0; i < n; i++) {
    if (x[i] != 0) {
      out->push_back(I[i]);
    }
  }
  return true;
}
