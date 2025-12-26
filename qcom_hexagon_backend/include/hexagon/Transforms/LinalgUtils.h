//===- LinalgUtils.h - Utils for working with linalg ops ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_LINALG_UTILS_H
#define HEXAGON_TRANSFORMS_LINALG_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"

/// Return a loop permutation for (m, n, k) that makes `n` innermost to
/// get unit‑stride access on row‑major B(k,n).
///  - AB:  {0,2,1}  // (m,k,n), n innermost
///  - ATB: {2,0,1}  // (k,m,n), n innermost
///  - else identity {0,1,2}
llvm::SmallVector<unsigned> getMatmulPermutation(mlir::linalg::LinalgOp op);

/// Return a loop permutation for (b, m, n, k) that makes `n` innermost to
/// get unit‑stride access on row‑major B(k,n).
///  - Batch AB:  {0,1,3,2}  // (b,m,k,n), n innermost
///  - Batch ATB: {0,3,1,2}  // (b,k,m,n), n innermost
///  - else identity {0,1,2,3}
llvm::SmallVector<unsigned>
getBatchMatmulPermutation(mlir::linalg::LinalgOp op);

#endif
