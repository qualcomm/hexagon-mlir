//===- LinalgUtils.cpp - Utils for working with linalg ops----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
