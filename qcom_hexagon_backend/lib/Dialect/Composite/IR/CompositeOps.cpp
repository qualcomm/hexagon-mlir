//===- CompositeOps.cpp - Composite op implementations --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Composite/IR/CompositeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::composite;

#define GET_OP_CLASSES
#include "hexagon/Dialect/Composite/IR/CompositeOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SoftmaxOp verification
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto initTy = cast<ShapedType>(getInit().getType());

  if (inputTy.getElementType() != initTy.getElementType())
    return emitOpError("input and init must have the same element type");

  if (inputTy.getRank() != initTy.getRank())
    return emitOpError("input and init must have the same rank");

  int64_t rank = inputTy.getRank();
  if (rank < 1)
    return emitOpError("input must have rank >= 1");

  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis must be in [0, rank), got ")
           << axis << " for rank " << rank;

  // In tensor mode, exactly one result with matching type.
  if (getNumResults() == 1) {
    auto resultTy = cast<ShapedType>(getResult()[0].getType());
    if (resultTy != initTy)
      return emitOpError("result type must match init type");
  } else if (getNumResults() > 1) {
    return emitOpError("expected 0 or 1 results");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RmsNormOp verification
//===----------------------------------------------------------------------===//

LogicalResult RmsNormOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto weightsTy = cast<ShapedType>(getWeights().getType());
  auto initTy = cast<ShapedType>(getInit().getType());

  if (inputTy.getElementType() != initTy.getElementType())
    return emitOpError("input and init must have the same element type");

  if (inputTy.getRank() != initTy.getRank())
    return emitOpError("input and init must have the same rank");

  int64_t rank = inputTy.getRank();
  if (rank < 1)
    return emitOpError("input must have rank >= 1");

  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis must be in [0, rank), got ")
           << axis << " for rank " << rank;

  if (weightsTy.getElementType() != inputTy.getElementType())
    return emitOpError("weights must have the same element type as input");

  if (getNumResults() == 1) {
    auto resultTy = cast<ShapedType>(getResult()[0].getType());
    if (resultTy != initTy)
      return emitOpError("result type must match init type");
  } else if (getNumResults() > 1) {
    return emitOpError("expected 0 or 1 results");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp — TilingInterface
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> SoftmaxOp::getLoopIteratorTypes() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  SmallVector<utils::IteratorType> iteratorTypes(inputTy.getRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getAxis()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> SoftmaxOp::getIterationDomain(OpBuilder &b) {
  auto inputTy = cast<ShapedType>(getInput().getType());
  int64_t rank = inputTy.getRank();
  Location loc = getLoc();

  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);

  SmallVector<Range> domain(rank);
  for (int64_t i = 0; i < rank; ++i) {
    Value size =
        inputTy.isDynamicDim(i)
            ? tensor::DimOp::create(b, loc, getInput(), i).getResult()
            : arith::ConstantIndexOp::create(b, loc, inputTy.getDimSize(i))
                  .getResult();
    domain[i] = Range{zero, size, one};
  }
  return domain;
}

FailureOr<TilingResult>
SoftmaxOp::getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = cast<ShapedType>(getInput().getType()).getRank();
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  Value inputSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInput(),
                                                    offsets, sizes, strides);
  Value initSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInit(),
                                                   offsets, sizes, strides);

  auto tiledOp = SoftmaxOp::create(b, getLoc(), initSlice.getType(),
                                   getAxisAttr(), inputSlice, initSlice);

  return TilingResult{{tiledOp}, {tiledOp.getResult()[0]}};
}

LogicalResult SoftmaxOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

FailureOr<TilingResult>
SoftmaxOp::generateResultTileValue(OpBuilder &b, unsigned resultNumber,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes);
}

//===----------------------------------------------------------------------===//
// RmsNormOp — TilingInterface
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> RmsNormOp::getLoopIteratorTypes() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  SmallVector<utils::IteratorType> iteratorTypes(inputTy.getRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getAxis()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> RmsNormOp::getIterationDomain(OpBuilder &b) {
  auto inputTy = cast<ShapedType>(getInput().getType());
  int64_t rank = inputTy.getRank();
  Location loc = getLoc();

  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);

  SmallVector<Range> domain(rank);
  for (int64_t i = 0; i < rank; ++i) {
    Value size =
        inputTy.isDynamicDim(i)
            ? tensor::DimOp::create(b, loc, getInput(), i).getResult()
            : arith::ConstantIndexOp::create(b, loc, inputTy.getDimSize(i))
                  .getResult();
    domain[i] = Range{zero, size, one};
  }
  return domain;
}

FailureOr<TilingResult>
RmsNormOp::getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = cast<ShapedType>(getInput().getType()).getRank();
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  Value inputSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInput(),
                                                    offsets, sizes, strides);
  Value weightsSlice = tensor::ExtractSliceOp::create(b, getLoc(), getWeights(),
                                                      offsets, sizes, strides);
  Value initSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInit(),
                                                   offsets, sizes, strides);

  auto tiledOp = RmsNormOp::create(
      b, getLoc(), initSlice.getType(), getAxisAttr(), getEpsilonAttr(),
      getNormSizeAttr(), inputSlice, weightsSlice, initSlice);

  return TilingResult{{tiledOp}, {tiledOp.getResult()[0]}};
}

LogicalResult RmsNormOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

FailureOr<TilingResult>
RmsNormOp::generateResultTileValue(OpBuilder &b, unsigned resultNumber,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes);
}

//===----------------------------------------------------------------------===//
// LayerNormOp verification
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto weightsTy = cast<ShapedType>(getWeights().getType());
  auto biasTy = cast<ShapedType>(getBias().getType());
  auto initTy = cast<ShapedType>(getInit().getType());

  if (inputTy.getElementType() != initTy.getElementType())
    return emitOpError("input and init must have the same element type");

  if (inputTy.getRank() != initTy.getRank())
    return emitOpError("input and init must have the same rank");

  int64_t rank = inputTy.getRank();
  if (rank < 1)
    return emitOpError("input must have rank >= 1");

  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis must be in [0, rank), got ")
           << axis << " for rank " << rank;

  if (weightsTy.getElementType() != inputTy.getElementType())
    return emitOpError("weights must have the same element type as input");

  if (biasTy.getElementType() != inputTy.getElementType())
    return emitOpError("bias must have the same element type as input");

  if (getNumResults() == 1) {
    auto resultTy = cast<ShapedType>(getResult()[0].getType());
    if (resultTy != initTy)
      return emitOpError("result type must match init type");
  } else if (getNumResults() > 1) {
    return emitOpError("expected 0 or 1 results");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp — TilingInterface
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> LayerNormOp::getLoopIteratorTypes() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  SmallVector<utils::IteratorType> iteratorTypes(inputTy.getRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getAxis()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> LayerNormOp::getIterationDomain(OpBuilder &b) {
  auto inputTy = cast<ShapedType>(getInput().getType());
  int64_t rank = inputTy.getRank();
  Location loc = getLoc();

  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);

  SmallVector<Range> domain(rank);
  for (int64_t i = 0; i < rank; ++i) {
    Value size =
        inputTy.isDynamicDim(i)
            ? tensor::DimOp::create(b, loc, getInput(), i).getResult()
            : arith::ConstantIndexOp::create(b, loc, inputTy.getDimSize(i))
                  .getResult();
    domain[i] = Range{zero, size, one};
  }
  return domain;
}

FailureOr<TilingResult>
LayerNormOp::getTiledImplementation(OpBuilder &b,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  int64_t rank = cast<ShapedType>(getInput().getType()).getRank();
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  Value inputSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInput(),
                                                    offsets, sizes, strides);
  Value weightsSlice = tensor::ExtractSliceOp::create(b, getLoc(), getWeights(),
                                                      offsets, sizes, strides);
  Value biasSlice = tensor::ExtractSliceOp::create(b, getLoc(), getBias(),
                                                   offsets, sizes, strides);
  Value initSlice = tensor::ExtractSliceOp::create(b, getLoc(), getInit(),
                                                   offsets, sizes, strides);

  // mean_out and rstd_out are memref side outputs, passed through unsliced.
  auto tiledOp = LayerNormOp::create(
      b, getLoc(), initSlice.getType(), getAxisAttr(), getEpsilonAttr(),
      getNormSizeAttr(), inputSlice, weightsSlice, biasSlice, initSlice,
      getMeanOut(), getRstdOut());

  return TilingResult{{tiledOp}, {tiledOp.getResult()[0]}};
}

LogicalResult LayerNormOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

FailureOr<TilingResult>
LayerNormOp::generateResultTileValue(OpBuilder &b, unsigned resultNumber,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes);
}
