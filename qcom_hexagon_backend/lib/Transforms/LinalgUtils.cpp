//===- LinalgUtils.cpp - Utils for working with linalg ops             ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Linalg level utils.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/LinalgUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

static bool matchMatmulMaps(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                            MLIRContext *ctx, ArrayRef<AffineExpr> aExprs,
                            ArrayRef<AffineExpr> bExprs,
                            ArrayRef<AffineExpr> cExprs, unsigned dims) {
  return aMap == AffineMap::get(dims, 0, aExprs, ctx) &&
         bMap == AffineMap::get(dims, 0, bExprs, ctx) &&
         cMap == AffineMap::get(dims, 0, cExprs, ctx);
}

static bool isMatmulAB(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                       MLIRContext *ctx) {
  AffineExpr m = getAffineDimExpr(0, ctx);
  AffineExpr n = getAffineDimExpr(1, ctx);
  AffineExpr k = getAffineDimExpr(2, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {m, k}, {k, n}, {m, n}, 3);
}

static bool isMatmulATB(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                        MLIRContext *ctx) {
  AffineExpr m = getAffineDimExpr(0, ctx);
  AffineExpr n = getAffineDimExpr(1, ctx);
  AffineExpr k = getAffineDimExpr(2, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {k, m}, {k, n}, {m, n}, 3);
}

static bool isMatmulABT(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                        MLIRContext *ctx) {
  AffineExpr m = getAffineDimExpr(0, ctx);
  AffineExpr n = getAffineDimExpr(1, ctx);
  AffineExpr k = getAffineDimExpr(2, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {m, k}, {n, k}, {m, n}, 3);
}

static bool isMatmulATBT(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                         MLIRContext *ctx) {
  AffineExpr m = getAffineDimExpr(0, ctx);
  AffineExpr n = getAffineDimExpr(1, ctx);
  AffineExpr k = getAffineDimExpr(2, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {k, m}, {n, k}, {m, n}, 3);
}

static bool isBatchMatmulAB(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                            MLIRContext *ctx) {
  AffineExpr b = getAffineDimExpr(0, ctx);
  AffineExpr m = getAffineDimExpr(1, ctx);
  AffineExpr n = getAffineDimExpr(2, ctx);
  AffineExpr k = getAffineDimExpr(3, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {b, m, k}, {b, k, n}, {b, m, n},
                         4);
}

static bool isBatchMatmulATB(AffineMap aMap, AffineMap bMap, AffineMap cMap,
                             MLIRContext *ctx) {
  AffineExpr b = getAffineDimExpr(0, ctx);
  AffineExpr m = getAffineDimExpr(1, ctx);
  AffineExpr n = getAffineDimExpr(2, ctx);
  AffineExpr k = getAffineDimExpr(3, ctx);
  return matchMatmulMaps(aMap, bMap, cMap, ctx, {b, k, m}, {b, k, n}, {b, m, n},
                         4);
}

llvm::SmallVector<unsigned> getMatmulPermutation(mlir::linalg::LinalgOp op) {
  auto maps = op.getIndexingMapsArray();

  if (maps.size() != 3)
    return {};

  // Canonical A·B
  if (isMatmulAB(maps[0], maps[1], maps[2], op.getContext()))
    return {0, 2, 1};

  // Aᵀ·B (A transposed, B canonical)
  if (isMatmulATB(maps[0], maps[1], maps[2], op.getContext()))
    return {2, 0, 1};

  // Identity
  return {0, 1, 2};
}

llvm::SmallVector<unsigned>
getBatchMatmulPermutation(mlir::linalg::LinalgOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return {};

  // A·B with batch
  if (isBatchMatmulAB(maps[0], maps[1], maps[2], op.getContext()))
    return {0, 1, 3, 2};

  // Aᵀ·B with batch
  if (isBatchMatmulATB(maps[0], maps[1], maps[2], op.getContext()))
    return {0, 3, 1, 2};

  // Identity
  return {0, 1, 2, 3};
}

bool isCanonicalRowMajorMatmul(mlir::linalg::LinalgOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  return isMatmulAB(maps[0], maps[1], maps[2], op.getContext());
}

MatmulMapOrientation getMatmulMapOrientation(mlir::linalg::LinalgOp op) {
  MatmulMapOrientation orient;
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return orient;
  MLIRContext *ctx = op.getContext();
  if (isMatmulAB(maps[0], maps[1], maps[2], ctx)) {
    orient.isMatmul = true;
    return orient;
  }
  if (isMatmulATB(maps[0], maps[1], maps[2], ctx)) {
    orient.isMatmul = true;
    orient.lhsTranspose = true;
    return orient;
  }
  if (isMatmulABT(maps[0], maps[1], maps[2], ctx)) {
    orient.isMatmul = true;
    orient.rhsTranspose = true;
    return orient;
  }
  if (isMatmulATBT(maps[0], maps[1], maps[2], ctx)) {
    orient.isMatmul = true;
    orient.lhsTranspose = true;
    orient.rhsTranspose = true;
    return orient;
  }
  return orient;
}

FailureOr<std::tuple<TypedValue<RankedTensorType>, TypedValue<RankedTensorType>,
                     TypedValue<RankedTensorType>>>
mlir::isHMXTypeAndRankCompatible(DestinationStyleOpInterface op,
                                 PatternRewriter &rewriter, const int64_t rank,
                                 bool allowU8I8, HMXMatmulKind *matchedKind) {
  Value a = op.getDpsInputOperand(0)->get();
  Value b = op.getDpsInputOperand(1)->get();
  Value c = op.getDpsInitOperand(0)->get();

  if (!a || !b || !c) {
    return rewriter.notifyMatchFailure(
        op, "must have two input operands, one init operand");
  }

  auto aType = dyn_cast<RankedTensorType>(a.getType());
  auto bType = dyn_cast<RankedTensorType>(b.getType());
  auto cType = dyn_cast<RankedTensorType>(c.getType());

  if (!aType || !bType || !cType) {
    return rewriter.notifyMatchFailure(op, "not ranked tensor types");
  }

  if (rank < 2 || rank > 3) {
    return rewriter.notifyMatchFailure(op, "inputs must be 2D or 3D");
  }

  if (aType.getRank() != rank || bType.getRank() != rank) {
    auto s = "tensors do not have rank " + std::to_string(rank);
    return rewriter.notifyMatchFailure(op, s);
  }

  Type aElem = aType.getElementType();
  Type bElem = bType.getElementType();
  Type cElem = cType.getElementType();

  // (f16, f16) -> f16|f32: the HMX f16 MAC.
  bool isF16 = aElem.isF16() && bElem.isF16();
  // (i8, i8) -> i32: HMX's mm_u8i8 reads the activation tile as unsigned and
  // the weight tile as signed. Both are signless i8 here; the caller is
  // responsible for bridging Triton's signed-dot semantics to the unsigned
  // activation the hardware wants.
  bool isU8I8 = allowU8I8 && aElem.isInteger(8) && bElem.isInteger(8) &&
                cElem.isInteger(32);

  if (!isF16 && !isU8I8) {
    return rewriter.notifyMatchFailure(
        op, allowU8I8 ? "inputs must be f16, or (i8, i8) with an i32 output"
                      : "inputs must be f16");
  }

  if (isF16 && !cElem.isF16() && !cElem.isF32()) {
    return rewriter.notifyMatchFailure(op, "output must be f16 or f32");
  }

  if (matchedKind)
    *matchedKind = isF16 ? HMXMatmulKind::F16 : HMXMatmulKind::U8I8;

  return std::tuple{cast<TypedValue<RankedTensorType>>(a),
                    cast<TypedValue<RankedTensorType>>(b),
                    cast<TypedValue<RankedTensorType>>(c)};
}
