//===-PreprocessTiledConv2D.cpp - preprocess tiled linalg.conv2d ops ------===//

//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass moves inverse layout transformation ops present outside tiled
// linalg.conv2d loops inside the loops. This will make it easier for later
// passes to lower linalg.conv2d to calls to HMX kernels for convolution
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/PackUnpackUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "preprocess-tiled-conv2d"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) (DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_PREPROCESSTILEDCONV2D
#include "hexagon/Transforms/Passes.h.inc"

namespace {

using CandidateTy = scf::ForOp;
using CandidatesTy = SmallVector<CandidateTy>;
using TiledLoopToConvOpMap =
    llvm::DenseMap<CandidateTy, linalg::Conv2DNhwcFhwcOp>;

using InputAndSliceOp = struct {
  std::optional<tensor::ExtractSliceOp> sliceOp;
  std::optional<Value> packedInput;
};

struct PreprocessTiledConv2DPass
    : public ::impl::PreprocessTiledConv2DBase<PreprocessTiledConv2DPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override;
  void populateCandidate(scf::ForOp op, FunctionOpInterface funcOp,
                         CandidatesTy &candidates, IRRewriter &rewriter);
  linalg::FillOp createAndInitPackedTensor(IRRewriter &rewriter,
                                           RankedTensorType RTT, Location loc);
  Value createPackedImageSlice(IRRewriter &rewriter, scf::ForOp oldTiledLoop,
                               scf::ForOp newLoop,
                               llvm::DenseMap<Value, Value> &loopVarsMap,
                               linalg::Conv2DNhwcFhwcOp convOp, Location &loc);
  Value createPackedFilterSlice(IRRewriter &rewriter, scf::ForOp oldTiledLoop,
                                scf::ForOp newLoop,
                                llvm::DenseMap<Value, Value> &loopVarsMap,
                                linalg::Conv2DNhwcFhwcOp convOp, Location &loc);
  void transformTiledConv2DLoop(CandidateTy oldTiledLoop, IRRewriter &rewriter);

  TiledLoopToConvOpMap loopToConvOp;
};

mlir::Value getConstVal(IRRewriter &rewriter, Location loc, int val) {
  IntegerAttr constant = rewriter.getIndexAttr(val);
  return rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  constant);
}

mlir::Value getOrCreateValue(IRRewriter &rewriter, Location loc,
                             OpFoldResult ofr, Type expectedType,
                             const llvm::DenseMap<Value, Value> &loopVarsMap) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    assert(intAttr);
    return getConstVal(rewriter, loc, intAttr.getInt());
  }
  Value val = ofr.dyn_cast<Value>();
  assert(val);
  auto itInMap = loopVarsMap.find(val);
  if (itInMap != loopVarsMap.end())
    return itInMap->second;
  return val;
}

void populatePackedImageSliceOffsets(
    IRRewriter &rewriter, Location loc,
    llvm::SmallVector<OpFoldResult> &offsets,
    llvm::SmallVector<Value> &packedSliceOffsets,
    llvm::DenseMap<Value, Value> &loopVarsMap) {
  assert(offsets.size() == 4);

  packedSliceOffsets.push_back(getOrCreateValue(
      rewriter, loc, offsets[0], rewriter.getIndexType(), loopVarsMap));
  mlir::Value numeric4DOffset1 = getOrCreateValue(
      rewriter, loc, offsets[1], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize1 =
      getConstVal(rewriter, loc, hexagon::F16_CROUTON_SHAPE[0]);
  packedSliceOffsets.push_back(rewriter.create<mlir::arith::FloorDivSIOp>(
      loc, rewriter.getIndexType(), numeric4DOffset1, numeric4DDimSize1));

  mlir::Value numeric4DOffset2 = getOrCreateValue(
      rewriter, loc, offsets[2], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize2 = getConstVal(
      rewriter, loc,
      hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3]);
  packedSliceOffsets.push_back(rewriter.create<mlir::arith::FloorDivSIOp>(
      loc, rewriter.getIndexType(), numeric4DOffset2, numeric4DDimSize2));

  mlir::Value numeric4DOffset3 = getOrCreateValue(
      rewriter, loc, offsets[3], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize3 =
      getConstVal(rewriter, loc, hexagon::F16_CROUTON_SHAPE[2]);
  packedSliceOffsets.push_back(rewriter.create<mlir::arith::FloorDivSIOp>(
      loc, rewriter.getIndexType(), numeric4DOffset3, numeric4DDimSize3));
  packedSliceOffsets.push_back(getConstVal(rewriter, loc, 0));
  packedSliceOffsets.push_back(packedSliceOffsets[4]);
  packedSliceOffsets.push_back(packedSliceOffsets[5]);
  packedSliceOffsets.push_back(packedSliceOffsets[6]);
}

void populatePackedImageSliceSizes(
    IRRewriter &rewriter, Location loc,
    llvm::SmallVector<int64_t> &unpackedSliceSizes,
    llvm::SmallVector<int64_t> &packedSliceSizes) {
  assert(unpackedSliceSizes.size() == 4);
  packedSliceSizes.push_back(unpackedSliceSizes[0]);
  packedSliceSizes.push_back(
      ceildiv(unpackedSliceSizes[1], hexagon::F16_CROUTON_SHAPE[0]));
  packedSliceSizes.push_back(
      ceildiv(unpackedSliceSizes[2],
              hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3]));
  packedSliceSizes.push_back(
      ceildiv(unpackedSliceSizes[3], hexagon::F16_CROUTON_SHAPE[2]));
  packedSliceSizes.push_back(hexagon::F16_CROUTON_SHAPE[0]);
  packedSliceSizes.push_back(hexagon::F16_CROUTON_SHAPE[1]);
  packedSliceSizes.push_back(hexagon::F16_CROUTON_SHAPE[2]);
  packedSliceSizes.push_back(hexagon::F16_CROUTON_SHAPE[3]);
}

Value getPackedImageSlice(IRRewriter &rewriter, Value packedImage, Location loc,
                          llvm::SmallVector<OpFoldResult> &offsets,
                          llvm::SmallVector<int64_t> &sizes,
                          llvm::SmallVector<int64_t> &strides,
                          llvm::DenseMap<Value, Value> &loopVarsMap) {
  // Extract the slice of packed image, corresponding to slice of conv2d op's
  // image

  llvm::SmallVector<Value> packedSliceOffsets;
  llvm::SmallVector<Value> emptyVecVals;
  llvm::SmallVector<int64_t> packedSliceSizesAsInt, packedSliceStridesAsInt,
      packedSliceOffsetsAsInt;

  populatePackedImageSliceOffsets(rewriter, loc, offsets, packedSliceOffsets,
                                  loopVarsMap);
  populatePackedImageSliceSizes(rewriter, loc, sizes, packedSliceSizesAsInt);
  Value one = getConstVal(rewriter, loc, 1);
  for (int i = 0; i < 8; ++i) {
    packedSliceStridesAsInt.push_back(1);
    packedSliceOffsetsAsInt.push_back(ShapedType::kDynamic);
  }
  auto packedImageRTT = llvm::cast<RankedTensorType>(packedImage.getType());

  RankedTensorType packedImageSliceRTT = RankedTensorType::get(
      packedSliceSizesAsInt, packedImageRTT.getElementType());
  return (rewriter.create<mlir::tensor::ExtractSliceOp>(
              loc, packedImageSliceRTT, packedImage, packedSliceOffsets,
              emptyVecVals, emptyVecVals, packedSliceOffsetsAsInt,
              packedSliceSizesAsInt, packedSliceStridesAsInt))
      .getResult();
}

void populatePackedFilterSliceOffsets(
    IRRewriter &rewriter, Location loc,
    llvm::SmallVector<OpFoldResult> &offsets,
    llvm::SmallVector<Value> &packedSliceOffsets,
    llvm::DenseMap<Value, Value> &loopVarsMap) {
  assert(offsets.size() == 4);

  // [F, FH, FW, FC] -> [F / 32, FC / 32, FH, FW, (FC / 2) % 16, F % 32, FC % 2]
  // Assumption: Only F and FC dims can be tiled.
  // F / 32
  Value outIndexVal = getOrCreateValue(rewriter, loc, offsets[0],
                                       rewriter.getIndexType(), loopVarsMap);
  packedSliceOffsets.push_back(rewriter.create<mlir::arith::FloorDivSIOp>(
      loc, rewriter.getIndexType(), outIndexVal,
      getConstVal(rewriter, loc, 32)));
  // FC / 32
  Value inIndexVal = getOrCreateValue(rewriter, loc, offsets[3],
                                      rewriter.getIndexType(), loopVarsMap);
  packedSliceOffsets.push_back(rewriter.create<mlir::arith::FloorDivSIOp>(
      loc, rewriter.getIndexType(), inIndexVal,
      getConstVal(rewriter, loc, 32)));
  // FH, FW dims
  packedSliceOffsets.push_back(getOrCreateValue(
      rewriter, loc, offsets[1], rewriter.getIndexType(), loopVarsMap));
  packedSliceOffsets.push_back(getOrCreateValue(
      rewriter, loc, offsets[2], rewriter.getIndexType(), loopVarsMap));
  // Tile dims
  Value constZero = getConstVal(rewriter, loc, 0);
  packedSliceOffsets.push_back(constZero);
  packedSliceOffsets.push_back(constZero);
  packedSliceOffsets.push_back(constZero);
}

void populatePackedFilterSliceSizes(
    IRRewriter &rewriter, Location loc,
    llvm::SmallVector<int64_t> &unpackedSliceSizes,
    llvm::SmallVector<int64_t> &packedSliceSizes) {
  assert(unpackedSliceSizes.size() == 4);
  packedSliceSizes.push_back(ceildiv(unpackedSliceSizes[0], 32));
  packedSliceSizes.push_back(ceildiv(unpackedSliceSizes[3], 32));
  packedSliceSizes.push_back(unpackedSliceSizes[1]);
  packedSliceSizes.push_back(unpackedSliceSizes[2]);
  packedSliceSizes.push_back(16);
  packedSliceSizes.push_back(32);
  packedSliceSizes.push_back(2);
}

Value getPackedFilterSlice(IRRewriter &rewriter, Value packedFilter,
                           Location loc,
                           llvm::SmallVector<OpFoldResult> &offsets,
                           llvm::SmallVector<int64_t> &sizes,
                           llvm::SmallVector<int64_t> &strides,
                           llvm::DenseMap<Value, Value> &loopVarsMap) {
  // Extract the slice of packed image, corresponding to slice of conv2d op's
  // image

  llvm::SmallVector<Value> packedSliceOffsets;
  llvm::SmallVector<Value> emptyVecVals;
  llvm::SmallVector<int64_t> packedSliceSizesAsInt, packedSliceStridesAsInt,
      packedSliceOffsetsAsInt;

  populatePackedFilterSliceOffsets(rewriter, loc, offsets, packedSliceOffsets,
                                   loopVarsMap);
  populatePackedFilterSliceSizes(rewriter, loc, sizes, packedSliceSizesAsInt);
  Value one = getConstVal(rewriter, loc, 1);
  for (int i = 0; i < 7; ++i) {
    packedSliceStridesAsInt.push_back(1);
    packedSliceOffsetsAsInt.push_back(ShapedType::kDynamic);
  }
  auto packedFilterRTT = llvm::cast<RankedTensorType>(packedFilter.getType());

  RankedTensorType packedFilterSliceRTT = RankedTensorType::get(
      packedSliceSizesAsInt, packedFilterRTT.getElementType());
  return (rewriter.create<mlir::tensor::ExtractSliceOp>(
              loc, packedFilterSliceRTT, packedFilter, packedSliceOffsets,
              emptyVecVals, emptyVecVals, packedSliceOffsetsAsInt,
              packedSliceSizesAsInt, packedSliceStridesAsInt))
      .getResult();
}

InputAndSliceOp getPackedInputAndSliceOp(linalg::Conv2DNhwcFhwcOp &conv2DOp,
                                         int argIndex) {
  InputAndSliceOp retVal;
  auto input = conv2DOp.getDpsInputOperand(argIndex)->get();
  auto extSliceOp = input.getDefiningOp<tensor::ExtractSliceOp>();

  // Check that the type of imageDefiningOp's input is packed F16 type of image.
  if (extSliceOp) {
    retVal.sliceOp = extSliceOp;
    Operation *dummyUnpackOp = retVal.sliceOp.value()
                                   .getSource()
                                   .getDefiningOp<UnrealizedConversionCastOp>();
    if (dummyUnpackOp) {
      retVal.packedInput = dummyUnpackOp->getOperand(0);
    }
  } else {
    if (auto dummyUnpackOp =
            input.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (dummyUnpackOp)
        retVal.packedInput = dummyUnpackOp.getOperand(0);
    }
  }
  return retVal;
}

std::optional<Value> getPackedInput(linalg::Conv2DNhwcFhwcOp &conv2DOp,
                                    int argIndex) {
  InputAndSliceOp inpAndSlice = getPackedInputAndSliceOp(conv2DOp, argIndex);
  return inpAndSlice.packedInput;
}

Value getPackedInputWithSliceInfo(linalg::Conv2DNhwcFhwcOp &conv2DOp,
                                  int argIndex,
                                  llvm::SmallVector<OpFoldResult> &offsets,
                                  llvm::SmallVector<int64_t> &sizes,
                                  llvm::SmallVector<int64_t> &strides) {
  InputAndSliceOp inpAndSlice = getPackedInputAndSliceOp(conv2DOp, argIndex);
  assert(inpAndSlice.packedInput);
  if (inpAndSlice.sliceOp) {
    DBG("ExtractSliceOp" << inpAndSlice.sliceOp << "\n");
    DBG("Offsets offsets: " << (inpAndSlice.sliceOp)->getStaticOffsets().size()
                            << "\n");
    DBG("Offsets sizes: " << (inpAndSlice.sliceOp)->getStaticSizes().size()
                          << "\n");
    DBG("Offsets strides: " << (inpAndSlice.sliceOp)->getStaticStrides().size()
                            << "\n");
    offsets = (inpAndSlice.sliceOp)->getMixedOffsets();
    auto staticSizes = (inpAndSlice.sliceOp)->getStaticSizes();
    sizes.assign(staticSizes.begin(), staticSizes.end());
    auto staticStrides = (inpAndSlice.sliceOp)->getStaticStrides();
    strides.assign(staticStrides.begin(), staticStrides.end());
  }
  return inpAndSlice.packedInput.value();
}

std::optional<Value> getPackedOutput(linalg::Conv2DNhwcFhwcOp &conv2DOp) {
  // Check that the conv2d output's only use is in an UnrealizedConversionCast
  // op that converts it to a packed F16 type.
  auto conv2DOutput = conv2DOp.getResult(0);
  if (!(conv2DOutput.hasOneUse()))
    return std::nullopt;
  auto insertSliceOp =
      dyn_cast<tensor::InsertSliceOp>((*(conv2DOutput.use_begin())).getOwner());
  if (!insertSliceOp)
    return std::nullopt;
  auto insertSliceResult = insertSliceOp.getResult();
  auto yieldOp =
      dyn_cast<scf::YieldOp>((*(insertSliceResult.use_begin())).getOwner());
  if (!yieldOp)
    return std::nullopt;
  auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  if (!forOp)
    return std::nullopt;
  auto result = forOp.getResult(0);
  if (auto dummyLayoutConversion = dyn_cast<UnrealizedConversionCastOp>(
          (*(result.use_begin())).getOwner()))
    return dummyLayoutConversion->getResult(0);

  return std::nullopt;
}

bool isWrappedConv2D(linalg::Conv2DNhwcFhwcOp &conv2DOp) {
  if (!isCandidate16BitElements(conv2DOp))
    return false;

  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<int64_t> sizes, strides;

  // Check that the type of imageDefiningOp's input is packed F16 type of image.
  auto dummyUnpackInput = getPackedInput(conv2DOp, 0);
  if (!(dummyUnpackInput.has_value()))
    return false;
  auto dummyUnpackImageInputType = dummyUnpackInput.value().getType();
  auto image = conv2DOp.getDpsInputOperand(0)->get();
  DBG("Image val: " << image << "\n");
  DBG("IMage Type: " << image.getType() << "\n");
  auto imageRTT = llvm::dyn_cast<RankedTensorType>(image.getType());
  if (!imageRTT)
    return false;

  auto dummyUnpackFilter = getPackedInput(conv2DOp, 1);
  if (!dummyUnpackFilter.has_value())
    return false;
  auto dummyUnpackFilterInputType = dummyUnpackFilter.value().getType();
  auto filter = conv2DOp.getDpsInputOperand(1)->get();
  auto filterRTT = llvm::dyn_cast<RankedTensorType>(filter.getType());
  if (!filterRTT)
    return false;

  // Check that the conv2d output's only use is in an UnrealizedConversionCast
  // op that converts it to a packed F16 type.
  auto dummyPackOutput = getPackedOutput(conv2DOp);
  if (!(dummyPackOutput.has_value()))
    return false;
  auto conv2DOutput = conv2DOp.getResult(0);
  if (auto outRTT = llvm::dyn_cast<RankedTensorType>(conv2DOutput.getType()))
    return true;
  return false;
}

/// Select for loops containing conv2d operations operating on tensor slices,
/// and yielding the full tensor after insertion of the tiled result.
/// The conv2d ops should be wrapped by inverse_pack and inverse_unpack ops.
void PreprocessTiledConv2DPass::populateCandidate(scf::ForOp op,
                                                  FunctionOpInterface funcOp,
                                                  CandidatesTy &candidates,
                                                  IRRewriter &rewriter) {
  DBG("Number of results:\n" << op.getNumResults() << "\n");
  DBG("For Op:\n" << op << "\n");

  if (op.getNumResults() != 1)
    return;
  Value forResult = op.getResult(0);
  DBG("forResult: " << forResult << "\nType: " << forResult.getType() << "\n");

  if (!isa<RankedTensorType>(forResult.getType()))
    return;

  auto *terminator = op.getBody()->getTerminator();
  auto yieldOp = llvm::cast<mlir::scf::YieldOp>(terminator);

  auto insertSliceOp =
      yieldOp.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSliceOp)
    return;
  // Check that the insert slice's source is a linalg.conv2d op
  auto wrappedConv2DOp =
      insertSliceOp.getSource().getDefiningOp<linalg::Conv2DNhwcFhwcOp>();
  if (!wrappedConv2DOp)
    return;
  if (!isWrappedConv2D(wrappedConv2DOp))
    return;

  DBG("selected candidate: " << op);
  candidates.emplace_back(op);
  loopToConvOp.insert({op, wrappedConv2DOp});
}

linalg::FillOp PreprocessTiledConv2DPass::createAndInitPackedTensor(
    IRRewriter &rewriter, RankedTensorType RTT, Location loc) {

  Value newOutTensor = rewriter.create<tensor::EmptyOp>(loc, RTT.getShape(),
                                                        RTT.getElementType());
  auto zeroF16Attr = rewriter.getF16FloatAttr(0.0);
  auto zeroF16 = rewriter.create<arith::ConstantOp>(loc, zeroF16Attr);
  return rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF16},
                                         ValueRange{newOutTensor});
}

Value PreprocessTiledConv2DPass::createPackedImageSlice(
    IRRewriter &rewriter, scf::ForOp oldTiledLoop, scf::ForOp newLoop,
    llvm::DenseMap<Value, Value> &loopVarsMap, linalg::Conv2DNhwcFhwcOp convOp,
    Location &loc) {

  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<int64_t> sizes, strides;
  auto packedInput =
      getPackedInputWithSliceInfo(convOp, 0, offsets, sizes, strides);
  if (offsets.size())
    return getPackedImageSlice(rewriter, packedInput, loc, offsets, sizes,
                               strides, loopVarsMap);
  return packedInput; // full packed input
}

Value PreprocessTiledConv2DPass::createPackedFilterSlice(
    IRRewriter &rewriter, scf::ForOp oldTiledLoop, scf::ForOp newLoop,
    llvm::DenseMap<Value, Value> &loopVarsMap, linalg::Conv2DNhwcFhwcOp convOp,
    Location &loc) {

  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<int64_t> sizes, strides;
  auto packedInput =
      getPackedInputWithSliceInfo(convOp, 1, offsets, sizes, strides);
  if (offsets.size())
    return getPackedFilterSlice(rewriter, packedInput, loc, offsets, sizes,
                                strides, loopVarsMap);
  return packedInput; // full packed filter
}

void getOutSlicePosition(linalg::Conv2DNhwcFhwcOp oldConvOp,
                         llvm::SmallVector<OpFoldResult> &offsets,
                         llvm::SmallVector<int64_t> &sizes,
                         llvm::SmallVector<int64_t> &strides) {
  // Find the insert slice op that inserts the output tile into full output
  // tensor.
  auto insertSliceOp = cast<tensor::InsertSliceOp>(
      (*(oldConvOp.getResult(0).use_begin())).getOwner());
  offsets = insertSliceOp.getMixedOffsets();
  auto staticSizes = insertSliceOp.getStaticSizes();
  sizes.assign(staticSizes.begin(), staticSizes.end());
  auto staticStrides = insertSliceOp.getStaticSizes();
  strides.assign(staticStrides.begin(), staticStrides.end());
}

tensor::InsertSliceOp
insertOutputSlice(IRRewriter &rewriter, Value packedFullOutput,
                  Value packedSliceOutput,
                  llvm::SmallVector<OpFoldResult> &unpackedSliceOffsets,
                  llvm::SmallVector<int64_t> &unpackedSliceSizes,
                  llvm::SmallVector<int64_t> &unpackedSliceStrides,
                  llvm::DenseMap<Value, Value> &loopVarsMap, Location &loc) {
  // Extract the slice of packed image, corresponding to slice of conv2d op's
  // image

  llvm::SmallVector<Value> packedSliceOffsets;
  llvm::SmallVector<Value> emptyVecVals;
  llvm::SmallVector<int64_t> packedSliceSizes, packedSliceStrides,
      packedSliceOffsetsAsInt;

  populatePackedImageSliceOffsets(rewriter, loc, unpackedSliceOffsets,
                                  packedSliceOffsets, loopVarsMap);
  populatePackedImageSliceSizes(rewriter, loc, unpackedSliceSizes,
                                packedSliceSizes);

  for (int i = 0; i < 8; ++i) {
    packedSliceStrides.push_back(1);
    packedSliceOffsetsAsInt.push_back(ShapedType::kDynamic);
  }
  auto packedFullOutputRTT =
      llvm::cast<RankedTensorType>(packedFullOutput.getType());

  return rewriter.create<mlir::tensor::InsertSliceOp>(
      loc, packedFullOutputRTT, packedSliceOutput, packedFullOutput,
      packedSliceOffsets, emptyVecVals, emptyVecVals, packedSliceOffsetsAsInt,
      packedSliceSizes, packedSliceStrides);
}

void PreprocessTiledConv2DPass::transformTiledConv2DLoop(
    CandidateTy oldTiledLoop, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(oldTiledLoop.getOperation());
  Location loc = oldTiledLoop.getLoc();
  Value oldLB = oldTiledLoop.getLowerBound();
  Value oldUB = oldTiledLoop.getUpperBound();
  Value oldStep = oldTiledLoop.getStep();

  // Create a new packed tensor, and initialize with zeros
  Value oldLoopResult = oldTiledLoop.getResult(0);
  auto oldResultRTT =
      mlir::cast<mlir::RankedTensorType>(oldLoopResult.getType());
  auto newResultRTT = getPacked16BitElementType(oldResultRTT);

  auto initOutOp = createAndInitPackedTensor(rewriter, newResultRTT, loc);

  // Create a new tiled loop over the new packed tensor.
  auto newLoop = rewriter.create<scf::ForOp>(loc, oldLB, oldUB, oldStep,
                                             initOutOp.getResult(0));

  rewriter.setInsertionPointToStart(newLoop.getBody());
  llvm::DenseMap<Value, Value> loopVarsMap;

  Value newCarried = newLoop.getRegionIterArgs().front();
  Value oldCarried = oldTiledLoop.getRegionIterArgs().front();
  loopVarsMap.insert({oldCarried, newCarried});
  loopVarsMap.insert(
      {oldTiledLoop.getInductionVar(), newLoop.getInductionVar()});

  // Get the linalg.conv2d op
  auto it = loopToConvOp.find(oldTiledLoop);
  assert(it != loopToConvOp.end());
  linalg::Conv2DNhwcFhwcOp convOp = cast<linalg::Conv2DNhwcFhwcOp>(it->second);

  // Create packed image slice
  auto packedImageSlice = createPackedImageSlice(
      rewriter, oldTiledLoop, newLoop, loopVarsMap, convOp, loc);
  DBG("Packed Image Slice: " << packedImageSlice << "\n");

  // Create packed filter slice
  auto packedFilterSlice = createPackedFilterSlice(
      rewriter, oldTiledLoop, newLoop, loopVarsMap, convOp, loc);
  DBG("Packed Filter Slice: " << packedFilterSlice << "\n");

  // create inverse pack of image and filter
  auto unpackedImageSlice = convOp.getDpsInputOperand(0)->get();
  auto unpackedImageSliceRTT =
      cast<RankedTensorType>(unpackedImageSlice.getType());
  auto inversePackedImageSlice = rewriter.create<UnrealizedConversionCastOp>(
      loc, mlir::TypeRange(unpackedImageSliceRTT),
      mlir::ValueRange({packedImageSlice}));

  auto unpackedFilterSlice = convOp.getDpsInputOperand(1)->get();
  auto unpackedFilterSliceRTT =
      cast<RankedTensorType>(unpackedFilterSlice.getType());
  auto inversePackedFilterSlice = rewriter.create<UnrealizedConversionCastOp>(
      loc, mlir::TypeRange(unpackedFilterSliceRTT),
      mlir::ValueRange({packedFilterSlice}));

  // Init output of new conv2d op, consuming unpackedImageSlice and
  // unpackedFilterslice
  auto unpackedOutputSlice = convOp.getOutputs().front();
  auto unpackedOutputSliceRTT =
      cast<RankedTensorType>(unpackedOutputSlice.getType());
  auto initSliceOutOp =
      createAndInitPackedTensor(rewriter, unpackedOutputSliceRTT, loc);

  // Create new conv2d op with unpackedImageSlice and unpackedFilterSlice
  // TODO: Copy attributes from old conv2d op to new conv2d op
  auto newConv2DSliceOp = rewriter.create<linalg::Conv2DNhwcFhwcOp>(
      loc,
      mlir::ValueRange({inversePackedImageSlice.getResult(0),
                        inversePackedFilterSlice.getResult(0)}),
      mlir::ValueRange({initSliceOutOp.getResult(0)}));

  // inverse unpack op on output of new conv2d op
  RankedTensorType inverseUnpackedSliceOutType =
      getPacked16BitElementType(unpackedOutputSliceRTT);
  auto inverseUnpackedSliceOut = rewriter.create<UnrealizedConversionCastOp>(
      loc, mlir::TypeRange(inverseUnpackedSliceOutType),
      mlir::ValueRange({newConv2DSliceOp.getResult(0)}));

  // insert this slice into full packed output (iter arg of for-loop)

  // first get unpacked output slice offsets, sizes and strides
  llvm::SmallVector<OpFoldResult> outSliceOffsets;
  llvm::SmallVector<int64_t> outSliceSizes, outSliceStrides;
  getOutSlicePosition(convOp, outSliceOffsets, outSliceSizes, outSliceStrides);

  // insert packed output slice into full packed output
  auto insertSliceOp = insertOutputSlice(
      rewriter, newCarried, inverseUnpackedSliceOut.getResult(0),
      outSliceOffsets, outSliceSizes, outSliceStrides, loopVarsMap, loc);

  rewriter.create<scf::YieldOp>(loc, insertSliceOp.getResult());
  rewriter.setInsertionPointAfter(newLoop);

  DBG("newTiledLoop: " << newLoop << "\n");
  DBG("Full Function:" << newLoop->getParentOfType<func::FuncOp>() << "\n");

  // The only use of old tiled loop's result is in a
  // builtin.unrealized_conversion_cast Replace uses of result of that
  // builtin.unrealized_conversion_cast with the result of the new for loop.
  assert(oldLoopResult.hasOneUse());
  auto inverseUnpackOp = cast<UnrealizedConversionCastOp>(
      (*(oldTiledLoop.getResult(0).use_begin())).getOwner());

  // Replace uses of inverseUnrealizedCastOp with the result of new tiled loop.
  inverseUnpackOp.getResult(0).replaceAllUsesWith(newLoop.getResult(0));
  rewriter.eraseOp(inverseUnpackOp);
  rewriter.eraseOp(oldTiledLoop);
}

void PreprocessTiledConv2DPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  CandidatesTy candidates;

  funcOp.walk([&](scf::ForOp op) {
    populateCandidate(op, funcOp, candidates, rewriter);
    return WalkResult::advance();
  });

  for (auto op : candidates)
    transformTiledConv2DLoop(op, rewriter);
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createPreprocessTiledConv2DPass() {
  return std::make_unique<PreprocessTiledConv2DPass>();
}
