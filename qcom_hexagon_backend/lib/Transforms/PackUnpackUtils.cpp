//===- PackUnpackUtils.cpp - Utils for generating pack/unpack ops----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#include "hexagon/Transforms/PackUnpackUtils.h"
#include "hexagon/Common/Common.h"
using namespace mlir;

// Utility to check if a RankedTensorType matches the expected element type and
// shape
bool hasValidTensorType(Value value, Type expectedType, int expectedRank) {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(value.getType());
  return tensorType && tensorType.getElementType() == expectedType &&
         tensorType.hasStaticShape() && tensorType.getRank() == expectedRank;
}

// Utility to check if strides and dilations are unit (i.e., [1, 1])
template <typename Conv2DOpT>
bool hasUnitStrideAndDilation(Conv2DOpT &conv2DOp) {
  auto strides = conv2DOp.getStrides().template getValues<int64_t>();
  auto dilations = conv2DOp.getDilations().template getValues<int64_t>();
  return strides.size() == 2 && strides[0] == 1 && strides[1] == 1 &&
         dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1;
}

// Generalized candidate type check for any Conv2D op and element type
template <typename Conv2DOpT>
bool isCandidateType(Conv2DOpT &conv2DOp, Type expectedType) {
  auto image = conv2DOp.getDpsInputOperand(0)->get();
  auto filter = conv2DOp.getDpsInputOperand(1)->get();
  return hasValidTensorType(image, expectedType) &&
         hasValidTensorType(filter, expectedType) &&
         hasUnitStrideAndDilation(conv2DOp);
}

// Checks if Conv2DNhwcFhwcOp is a candidate for wrapping with layout
// conversion ops.
bool isCandidate16BitElements(linalg::Conv2DNhwcFhwcOp &conv2DOp) {
  return isCandidateType(conv2DOp, Float16Type::get(conv2DOp.getContext()));
}

// Checks if Conv2DNhwcHwcfQOp is a candidate for wrapping with layout
// conversion ops.
bool isCandidate8BitElements(linalg::Conv2DNhwcHwcfQOp &conv2DOp) {
  return isCandidateType(conv2DOp, IntegerType::get(conv2DOp.getContext(), 8));
}

int ceildiv(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

// FP16 layout conversions for images (data) require 2 pack ops.
// The first one effects this mapping: (n, h, w, c) -> (n, h/8, w/4, c/32, 8, 4,
// 32) This function returns the tensor type of the first pack.
RankedTensorType
getTempPacked16BitElementType(const RankedTensorType &origType) {
  auto origShape = origType.getShape();
  assert(origShape.size() == 4);
  SmallVector<int64_t> paddedShape = {
      origShape[0],
      ceildiv(origShape[1], hexagon::F16_CROUTON_SHAPE[0]),
      ceildiv(origShape[2],
              hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3]),
      ceildiv(origShape[3], hexagon::F16_CROUTON_SHAPE[2]),
      hexagon::F16_CROUTON_SHAPE[0],
      hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3],
      hexagon::F16_CROUTON_SHAPE[2]};
  return RankedTensorType::get(paddedShape, origType.getElementType());
}

// This function returns the FP16-layout-converted tensor's type
// For an (n, h, w, c) tensor type it returns (n, h/8, w/4, c/32, 8, 2, 32, 2)
RankedTensorType getPacked16BitElementType(const RankedTensorType &origType) {
  auto origShape = origType.getShape();
  assert(origShape.size() == 4);
  SmallVector<int64_t> paddedShape = {
      origShape[0],
      ceildiv(origShape[1], hexagon::F16_CROUTON_SHAPE[0]),
      ceildiv(origShape[2],
              hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3]),
      ceildiv(origShape[3], hexagon::F16_CROUTON_SHAPE[2]),
      hexagon::F16_CROUTON_SHAPE[0],
      hexagon::F16_CROUTON_SHAPE[1],
      hexagon::F16_CROUTON_SHAPE[2],
      hexagon::F16_CROUTON_SHAPE[3]};
  return RankedTensorType::get(paddedShape, origType.getElementType());
}

// This function returns the FP16-layout-converted filter's type
// For an (f, h, w, c) filter it returns (f/32, c/32, h, w, 16, 32, 2)
RankedTensorType
getPackedFilter16BitElementType(const RankedTensorType &origType) {
  auto origShape = origType.getShape();
  assert(origShape.size() == 4);
  SmallVector<int64_t> paddedShape = {ceildiv(origShape[0], 32),
                                      ceildiv(origShape[3], 32),
                                      origShape[1],
                                      origShape[2],
                                      16,
                                      32,
                                      2};
  return RankedTensorType::get(paddedShape, origType.getElementType());
}

// This function returns the INT8-layout-converted tensor's type
// For an (n, h, w, c) tensor type it returns (n, h/8, w/8, c/32, 8, 8, 32)
RankedTensorType getPacked8BitElementType(const RankedTensorType &origType) {
  auto origShape = origType.getShape();
  assert(origShape.size() == 4);
  SmallVector<int64_t> paddedShape = {
      origShape[0],
      ceildiv(origShape[1], hexagon::INT8_CROUTON_SHAPE[0]),
      ceildiv(origShape[2], hexagon::INT8_CROUTON_SHAPE[1]),
      ceildiv(origShape[3], hexagon::INT8_CROUTON_SHAPE[2]),
      hexagon::INT8_CROUTON_SHAPE[0],
      hexagon::INT8_CROUTON_SHAPE[1],
      hexagon::INT8_CROUTON_SHAPE[2]};
  return RankedTensorType::get(paddedShape, origType.getElementType());
}

static mlir::linalg::PackOp
createPackOp(RewriterBase &rewriter, Location &loc, Value input,
             RankedTensorType &packedType, SmallVector<int64_t> &dimsPos,
             SmallVector<OpFoldResult> &tiles, Value padValue) {
  Value emptyPackedImage = tensor::EmptyOp::create(
      rewriter, loc, packedType.getShape(), packedType.getElementType());
  return linalg::PackOp::create(rewriter, loc, input, emptyPackedImage, dimsPos,
                                tiles, padValue);
}

static mlir::linalg::UnPackOp createUnpackOp(RewriterBase &rewriter,
                                             Location &loc, Value input,
                                             RankedTensorType &inputType,
                                             ArrayRef<int64_t> &origShape,
                                             SmallVector<int64_t> &dimsPos,
                                             SmallVector<OpFoldResult> &tiles) {

  Value emptyOrigImage = tensor::EmptyOp::create(rewriter, loc, origShape,
                                                 inputType.getElementType());
  return linalg::UnPackOp::create(rewriter, loc, input, emptyOrigImage, dimsPos,
                                  tiles);
}

// This helper function generates code for layout conversions for an i8
// tensor.
linalg::PackOp buildI8PackOps(RewriterBase &rewriter, Value image,
                              RankedTensorType &imageRTType, Location &loc,
                              Value padValue,
                              SmallVector<int64_t> &innerDimsPos,
                              SmallVector<OpFoldResult> &innerTiles) {
  RankedTensorType packedI8Type = getPacked8BitElementType(imageRTType);
  auto packOp = createPackOp(rewriter, loc, image, packedI8Type, innerDimsPos,
                             innerTiles, padValue);
  return packOp;
}

// Helper function for unpacking the output of a I8 op.
linalg::UnPackOp unpackI8Output(RewriterBase &rewriter, Value unpackInput,
                                Location &loc,
                                SmallVector<int64_t> &innerDimsPos,
                                SmallVector<OpFoldResult> &innerTiles,
                                RankedTensorType &outRTTy,
                                ArrayRef<int64_t> &outShape) {
  return createUnpackOp(rewriter, loc, unpackInput, outRTTy, outShape,
                        innerDimsPos, innerTiles);
}

// This helper function generates code for layout conversions for an FP16
// tensor.
linalg::PackOp buildF16PackOps(RewriterBase &rewriter, Value image,
                               RankedTensorType &imageRTType, Location &loc,
                               mlir::Value padValue,
                               SmallVector<int64_t> &innerDimsPosTemp,
                               SmallVector<OpFoldResult> &innerTilesTemp,
                               SmallVector<int64_t> &innerDimsPos,
                               SmallVector<OpFoldResult> &innerTiles) {

  RankedTensorType tempPackedF16Type =
      getTempPacked16BitElementType(imageRTType);
  RankedTensorType packedF16Type = getPacked16BitElementType(imageRTType);

  auto packToTempOp = createPackOp(rewriter, loc, image, tempPackedF16Type,
                                   innerDimsPosTemp, innerTilesTemp, padValue);
  auto packOp = createPackOp(rewriter, loc, packToTempOp.getResult(),
                             packedF16Type, innerDimsPos, innerTiles, Value{});
  return packOp;
}

// Helper function for unpacking the output of a packed FP16 op.
linalg::UnPackOp unpackF16Output(RewriterBase &rewriter, Value firstUnpackInput,
                                 Location &loc,
                                 SmallVector<int64_t> &innerDimsPos,
                                 SmallVector<OpFoldResult> &innerTiles,
                                 SmallVector<int64_t> &innerDimsPosTemp,
                                 SmallVector<OpFoldResult> &innerTilesTemp,
                                 RankedTensorType &outRTTy,
                                 ArrayRef<int64_t> &outShape) {

  RankedTensorType tempPackedF16Type = getTempPacked16BitElementType(outRTTy);
  auto tempShape = tempPackedF16Type.getShape();
  auto tmpUnpackOp =
      createUnpackOp(rewriter, loc, firstUnpackInput, tempPackedF16Type,
                     tempShape, innerDimsPos, innerTiles);

  auto unpackOp =
      createUnpackOp(rewriter, loc, tmpUnpackOp.getResult(), outRTTy, outShape,
                     innerDimsPosTemp, innerTilesTemp);
  return unpackOp;
}
