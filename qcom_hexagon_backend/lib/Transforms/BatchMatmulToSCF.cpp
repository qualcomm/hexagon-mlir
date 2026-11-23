//===- BatchMatmulToSCF.cpp - Pattern to map batch matmul to HMX ops ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/BatchMatmulToSCF.h"
#include "hexagon/Transforms/LinalgUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "batch-matmul-to-scf-hmx"

namespace mlir {

TypedValue<RankedTensorType>
LoopedHMXMatmul::extractSlice(PatternRewriter &rewriter, Location loc,
                              TypedValue<IndexType> select,
                              TypedValue<RankedTensorType> tensor) const {
  SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets = {select, rewriter.getIndexAttr(0),
                                       rewriter.getIndexAttr(0)};

  auto ttype = tensor.getType();
  auto m = ttype.getShape()[1];
  auto k = ttype.getShape()[2];

  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(m),
                                     rewriter.getIndexAttr(k)};

  auto etype = RankedTensorType::get({m, k}, ttype.getElementType());

  return tensor::ExtractSliceOp::create(rewriter, loc, etype, tensor, offsets,
                                        sizes, strides)
      .getResult();
}

TypedValue<RankedTensorType>
LoopedHMXMatmul::insertSlice(PatternRewriter &rewriter, Location loc,
                             TypedValue<IndexType> select,
                             TypedValue<RankedTensorType> source,
                             TypedValue<RankedTensorType> dest) const {
  SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets = {select, rewriter.getIndexAttr(0),
                                       rewriter.getIndexAttr(0)};

  auto ttype = source.getType();
  auto m = ttype.getShape()[0];
  auto k = ttype.getShape()[1];

  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(m),
                                     rewriter.getIndexAttr(k)};

  return tensor::InsertSliceOp::create(rewriter, loc, source, dest, offsets,
                                       sizes, strides)
      .getResult();
}

LogicalResult
LoopedHMXMatmul::matchAndRewrite(linalg::BatchMatmulOp op,
                                 PatternRewriter &rewriter) const {

  const auto rank = 3;
  auto check = isHMXTypeAndRankCompatible(op, rewriter, rank);
  if (failed(check))
    return failure();

  auto [a, b, c] = *check;

  LLVM_DEBUG({
    llvm::dbgs()
        << "[" << DEBUG_TYPE
        << "] Converting batch matmul to scf.for { matmul } for HMX :\n";
    llvm::dbgs() << "  LHS: " << a << "\n";
    llvm::dbgs() << "  RHS: " << b << "\n";
    llvm::dbgs() << "  Output: " << c << "\n";
  });

  int64_t s = a.getType().getShape()[0];
  Location loc = op.getLoc();
  auto zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
  auto step =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
  auto upper =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(s));

  auto scf = scf::ForOp::create(rewriter, loc, zero, upper, step, c);
  rewriter.setInsertionPointToStart(scf.getBody());

  auto idx = cast<TypedValue<IndexType>>(scf.getInductionVar());
  auto iterArg =
      cast<TypedValue<RankedTensorType>>(scf.getBody()->getArgument(1));

  auto extractA = extractSlice(rewriter, loc, idx, a);
  auto extractB = extractSlice(rewriter, loc, idx, b);
  auto extractC = extractSlice(rewriter, loc, idx, iterArg);

  auto mul =
      linalg::MatmulOp::create(rewriter, loc, {extractA, extractB}, extractC);

  auto res = cast<TypedValue<RankedTensorType>>(mul->getResult(0));

  auto updated = insertSlice(rewriter, loc, idx, res, iterArg);

  scf::YieldOp::create(rewriter, loc, updated);

  rewriter.replaceOp(op, scf.getResult(0));

  return success();
}

} // namespace mlir
