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

/// Returns true iff `op` is a canonical row-major matmul:
/// A(m,k) * B(k,n) -> C(m,n) with the standard indexing maps and no
/// transpose / broadcast applied via the optional indexing_maps attribute.
bool isCanonicalRowMajorMatmul(mlir::linalg::LinalgOp op);

/// Describes the orientation of a 2-D `linalg::MatmulOp` (or compatible
/// `linalg::GenericOp`) by inspecting its `indexing_maps`.  Identifies one
/// of the four supported orientations:
///
///   isMatmul   lhsTranspose   rhsTranspose   pattern
///   --------   ------------   ------------   -------
///   true       false          false          C(m,n) = A(m,k) · B(k,n)
///   true       true           false          C(m,n) = A(k,m)ᵀ · B(k,n)
///   true       false          true           C(m,n) = A(m,k) · B(n,k)ᵀ
///   true       true           true           C(m,n) = A(k,m)ᵀ · B(n,k)ᵀ
///   false      *              *              (non-matmul / unsupported)
///
/// The C map is always identity `(m,n) -> (m,n)`.  In all cases the iteration
/// space is (m, n, k); only the operand access patterns change.  Returns
/// `isMatmul = false` for any other shape (broadcasts, higher rank, etc.).
struct MatmulMapOrientation {
  bool isMatmul = false;
  bool lhsTranspose = false;
  bool rhsTranspose = false;
};
MatmulMapOrientation getMatmulMapOrientation(mlir::linalg::LinalgOp op);

namespace mlir {
/// Which HMX matmul element-type contract an operation matched.
///   F16  : (f16, f16) -> f16|f32   -- HMX f16 MAC.
///   U8I8 : (i8,  i8 ) -> i32       -- HMX mixed unsigned-activation /
///                                     signed-weight MAC (mm_u8i8). Both
///                                     operands are signless `i8`; "u8
///                                     activation" is a contract carried by
///                                     operand position, not by the MLIR type,
///                                     matching the hexkl dialect's micro ops.
enum class HMXMatmulKind { F16, U8I8 };

/// Returns the typed operand tensors to the batched matmul only if the
/// operation maps to HMX.
///
/// `allowU8I8` additionally admits the (i8, i8) -> i32 contract; it defaults to
/// false so existing callers keep the f16-only behaviour. Callers that opt in
/// must handle both kinds -- pass `matchedKind` to learn which one matched.
FailureOr<std::tuple<TypedValue<RankedTensorType>, TypedValue<RankedTensorType>,
                     TypedValue<RankedTensorType>>>
isHMXTypeAndRankCompatible(DestinationStyleOpInterface op,
                           PatternRewriter &rewriter, const int64_t rank,
                           bool allowU8I8 = false,
                           HMXMatmulKind *matchedKind = nullptr);

} // namespace mlir

#endif
