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
// linalg.conv2d loops inside the loops, and make tiling loops for linalg.conv2d
// return packed tensors. This will make it easier for later
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

using CandidateLoopTy = scf::ForOp;
using CandidateLoopsTy = SmallVector<CandidateLoopTy>;
using CandidateConvTy = linalg::Conv2DNhwcFhwcOp;
using CandidateConvsTy = SmallVector<CandidateConvTy>;
using TiledLoopToConvOpMap =
    llvm::DenseMap<CandidateLoopTy, linalg::Conv2DNhwcFhwcOp>;
using ProcessedConvMap = llvm::DenseMap<linalg::Conv2DNhwcFhwcOp, bool>;
using InputAndSliceOp = struct {
  std::optional<tensor::ExtractSliceOp> sliceOp;
  std::optional<Value> packedInput;
};

struct PreprocessTiledConv2DPass
    : public ::impl::PreprocessTiledConv2DBase<PreprocessTiledConv2DPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override;
  // identify scf.for loopnests performing linalg.conv2d ops on
  // input tiles and populate them in 'candidates'. Also keep track
  // of the conv2d ops inside candidate loops in 'processedConvs'
  bool populateCandidateLoops(scf::ForOp op, CandidateLoopsTy &candidates,
                              ProcessedConvMap &processedConvs,
                              IRRewriter &rewriter);
  // Creates a zero-initialized tensor of given type
  Value createAndInitTensor(IRRewriter &rewriter, RankedTensorType RTT,
                            Location loc);
  // Given a conv2d op whose flat inputs are derived from a packed tensor,
  // (through a sequence of inverse_pack op and possibly extract_slice),
  // this function returns the packed tensor slice mapping to the unpacked
  // input of the conv2d op. The returned slice (or full tensor) could be
  // bigger than the input.
  Value createPackedImageSlice(IRRewriter &rewriter,
                               const llvm::DenseMap<Value, Value> &loopVarsMap,
                               linalg::Conv2DNhwcFhwcOp convOp, Location &loc);
  // Given a conv2d op whose flat filter is derived from a packed tensor,
  // (through a sequence of inverse_pack op and possibly extract_slice),
  // this function returns the packed filter slice mapping to the unpacked
  // input of the conv2d op. The returned slice (or full tensor) could be
  // bigger than the input.
  Value createPackedFilterSlice(IRRewriter &rewriter,
                                const llvm::DenseMap<Value, Value> &loopVarsMap,
                                linalg::Conv2DNhwcFhwcOp convOp, Location &loc);

  // Moves inverse layout transformation ops present outside tiled
  // linalg.conv2d loops inside the loops, making tiling loops for linalg.conv2d
  // return packed tensors. Returns the new loop after this transformation.
  // If inArg is not null, it should be produced by an inverse pack op, and
  // the inArg for the new loop is the source of the inverse pack op (a packed
  // tensor). parentMap maps variables in old parent loop to the corresponding
  // variables in the new parent loops.
  scf::ForOp
  transformTiledConv2DLoop(CandidateLoopTy oldTiledLoop, IRRewriter &rewriter,
                           Value inArg = nullptr,
                           const llvm::DenseMap<Value, Value> &parentMap =
                               llvm::DenseMap<Value, Value>());

  // Transforms a linalg.conv2d op consuming slices from unpacked tensors
  // to a new linalg.conv2d op, consuming slices from the corresponding packed
  // tensors, and unpacked before getting consumed by the new linalg.conv2d op.
  void transformWrappedConv2DOp(linalg::Conv2DNhwcFhwcOp conv2DOp,
                                IRRewriter &rewriter);

  // Helper for transforming linalg.conv2d op.
  tensor::InsertSliceOp transformWrappedConv2DOpHelper(
      IRRewriter &rewriter, linalg::Conv2DNhwcFhwcOp convOp,
      Value packedPartialOutput,
      const llvm::DenseMap<Value, Value> &loopVarsMap, Location &loc);

  // Maps the innermost of a tiled loop to the linalg.conv2d op it tiles
  TiledLoopToConvOpMap loopToConvOp;
};

mlir::Value getConstVal(IRRewriter &rewriter, Location loc, int val) {
  IntegerAttr constant = rewriter.getIndexAttr(val);
  return mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                         constant);
}

// Given an OpFoldResult object ofr, this function returns
// the constant if the object holds a constant OR
// the loopVarsMap[ofr.value] if ofr.value is in loopVarsMap, OR
// ofr.value otherwise
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

// Populates offset expressions for the packed image slice, given offsets in
// an unpacked image slice and the map from variables in offset expressions
// for unpacked image to the variables in offset expressions for
// packed image.
void populatePackedImageSliceOffsets(
    IRRewriter &rewriter, Location loc,
    const llvm::SmallVector<OpFoldResult> &offsets,
    llvm::SmallVector<Value> &packedSliceOffsets,
    const llvm::DenseMap<Value, Value> &loopVarsMap) {
  assert(offsets.size() == 4);

  packedSliceOffsets.push_back(getOrCreateValue(
      rewriter, loc, offsets[0], rewriter.getIndexType(), loopVarsMap));
  mlir::Value numeric4DOffset1 = getOrCreateValue(
      rewriter, loc, offsets[1], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize1 =
      getConstVal(rewriter, loc, hexagon::F16_CROUTON_SHAPE[0]);
  packedSliceOffsets.push_back(
      mlir::arith::FloorDivSIOp::create(rewriter, loc, rewriter.getIndexType(),
                                        numeric4DOffset1, numeric4DDimSize1));

  mlir::Value numeric4DOffset2 = getOrCreateValue(
      rewriter, loc, offsets[2], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize2 = getConstVal(
      rewriter, loc,
      hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3]);
  packedSliceOffsets.push_back(
      mlir::arith::FloorDivSIOp::create(rewriter, loc, rewriter.getIndexType(),
                                        numeric4DOffset2, numeric4DDimSize2));

  mlir::Value numeric4DOffset3 = getOrCreateValue(
      rewriter, loc, offsets[3], rewriter.getIndexType(), loopVarsMap);
  mlir::Value numeric4DDimSize3 =
      getConstVal(rewriter, loc, hexagon::F16_CROUTON_SHAPE[2]);
  packedSliceOffsets.push_back(
      mlir::arith::FloorDivSIOp::create(rewriter, loc, rewriter.getIndexType(),
                                        numeric4DOffset3, numeric4DDimSize3));
  packedSliceOffsets.push_back(getConstVal(rewriter, loc, 0));
  packedSliceOffsets.push_back(packedSliceOffsets[4]);
  packedSliceOffsets.push_back(packedSliceOffsets[5]);
  packedSliceOffsets.push_back(packedSliceOffsets[6]);
}

// Populates sizes for the packed image slice, given sizes in
// an unpacked image slice.
void populatePackedImageSliceSizes(
    IRRewriter &rewriter, Location loc,
    const llvm::SmallVector<int64_t> &unpackedSliceSizes,
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

// Creates and returns the slice of a packed image, given the offsets, sizes
// and strides of unpacked image and the map from variables used in
// offset expressions for unpacked image to the variables to be used
// in the offset expressions for the packed image.
Value getPackedImageSlice(IRRewriter &rewriter, Value packedImage, Location loc,
                          const llvm::SmallVector<OpFoldResult> &offsets,
                          const llvm::SmallVector<int64_t> &sizes,
                          const llvm::SmallVector<int64_t> &strides,
                          const llvm::DenseMap<Value, Value> &loopVarsMap) {

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
  auto packedImageRTT = cast<RankedTensorType>(packedImage.getType());

  RankedTensorType packedImageSliceRTT = RankedTensorType::get(
      packedSliceSizesAsInt, packedImageRTT.getElementType());
  return (mlir::tensor::ExtractSliceOp::create(
              rewriter, loc, packedImageSliceRTT, packedImage,
              packedSliceOffsets, emptyVecVals, emptyVecVals,
              packedSliceOffsetsAsInt, packedSliceSizesAsInt,
              packedSliceStridesAsInt))
      .getResult();
}

// Populates offset expressions for the packed filter slice, given offsets in
// an unpacked filter slice and the map from variables in offset expressions
// for unpacked filter to the variables in offset expressions for
// packed filter.
void populatePackedFilterSliceOffsets(
    IRRewriter &rewriter, Location loc,
    llvm::SmallVector<OpFoldResult> &offsets,
    llvm::SmallVector<Value> &packedSliceOffsets,
    const llvm::DenseMap<Value, Value> &loopVarsMap) {
  assert(offsets.size() == 4);

  // [F, FH, FW, FC] -> [F / 32, FC / 32, FH, FW, (FC / 2) % 16, F % 32, FC % 2]
  // Assumption: Only F and FC dims can be tiled.
  Value outIndexVal = getOrCreateValue(rewriter, loc, offsets[0],
                                       rewriter.getIndexType(), loopVarsMap);
  packedSliceOffsets.push_back(mlir::arith::FloorDivSIOp::create(
      rewriter, loc, rewriter.getIndexType(), outIndexVal,
      getConstVal(rewriter, loc, 32)));
  // FC / 32
  Value inIndexVal = getOrCreateValue(rewriter, loc, offsets[3],
                                      rewriter.getIndexType(), loopVarsMap);
  packedSliceOffsets.push_back(mlir::arith::FloorDivSIOp::create(
      rewriter, loc, rewriter.getIndexType(), inIndexVal,
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

// Populates sizes for the packed filter slice, given sizes in
// an unpacked filter slice.
void populatePackedFilterSliceSizes(
    IRRewriter &rewriter, Location loc,
    const llvm::SmallVector<int64_t> &unpackedSliceSizes,
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

// Creates and returns the slice of a packed filter, given the offsets, sizes
// and strides of unpacked filter and the map from variables used in
// offset expressions for unpacked filter to the variables to be used
// in the offset expressions for the packed filter.
Value getPackedFilterSlice(IRRewriter &rewriter, Value packedFilter,
                           Location loc,
                           llvm::SmallVector<OpFoldResult> &offsets,
                           llvm::SmallVector<int64_t> &sizes,
                           llvm::SmallVector<int64_t> &strides,
                           const llvm::DenseMap<Value, Value> &loopVarsMap) {

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
  auto packedFilterRTT = cast<RankedTensorType>(packedFilter.getType());

  RankedTensorType packedFilterSliceRTT = RankedTensorType::get(
      packedSliceSizesAsInt, packedFilterRTT.getElementType());
  return (mlir::tensor::ExtractSliceOp::create(
              rewriter, loc, packedFilterSliceRTT, packedFilter,
              packedSliceOffsets,
              // passing static sizes & strides, so empty value vectors for them
              emptyVecVals, emptyVecVals,
              // static values for offsets, sizes and strides
              packedSliceOffsetsAsInt, packedSliceSizesAsInt,
              packedSliceStridesAsInt))
      .getResult();
}

// Given a linalg::Conv2DNhwcFhwcOp and an arg index, returns a struct
// containing the packed input tensor and the slice op that extracts it.
// In particular, this is the function's expectation:
// unpackedIn = builtin.unrealized_conversion_cast(packedIn, UnpackedInType)
// slice = tensor.extract_slice(unpackedIn, offsets, sizes, strides)
// convResult = linalg.conv2d_nhwc_fhwc(slice, ...)
// The function returns packedIn and the extract_slice op, if present.
InputAndSliceOp getPackedInputAndSliceOp(linalg::Conv2DNhwcFhwcOp &conv2DOp,
                                         int argIndex) {
  InputAndSliceOp retVal;
  auto input = conv2DOp.getDpsInputOperand(argIndex)->get();
  auto extSliceOp = input.getDefiningOp<tensor::ExtractSliceOp>();

  if (extSliceOp) {
    retVal.sliceOp = extSliceOp;
    Operation *dummyUnpackOp =
        extSliceOp.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (dummyUnpackOp) {
      retVal.packedInput = dummyUnpackOp->getOperand(0);
    }
  } else {
    // full tensor (not slice)
    if (auto dummyUnpackOp =
            input.getDefiningOp<UnrealizedConversionCastOp>()) {
      retVal.packedInput = dummyUnpackOp.getOperand(0);
    }
  }
  return retVal;
}

// Given a linalg::Conv2DNhwcFhwcOp and an arg index, returns the packed input
// Index could correspond to image or filter
std::optional<Value> getPackedInput(linalg::Conv2DNhwcFhwcOp &conv2DOp,
                                    int argIndex) {
  InputAndSliceOp inpAndSlice = getPackedInputAndSliceOp(conv2DOp, argIndex);
  return inpAndSlice.packedInput;
}

// Given a linalg::Conv2DNhwcFhwcOp and an arg index, returns the packed
// input from which the full arg-index-th input is derived, and the slice
// info (offsets, sizes, strides) of the extract_slice op that produces
// the conv2d op's arg-index-th input. If the input is not sliced, then
// slice info will be empty.
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

// Given a linalg::Conv2DNhwcFhwcOp, returns the packed output value.
// Expectation: Its output is used only in an insert-slice op, whose
// only use is an UnrealizedConversionCast op that packs it.
std::optional<Value> getPackedOutput(linalg::Conv2DNhwcFhwcOp &conv2DOp) {
  // Check that the conv2d output's only use is in an UnrealizedConversionCast
  // op that converts it to a packed F16 type.
  auto conv2DOutput = conv2DOp.getResult(0);
  if (!(conv2DOutput.hasOneUse()))
    return std::nullopt;
  auto insertSliceOp =
      dyn_cast<tensor::InsertSliceOp>((*(conv2DOutput.use_begin())).getOwner());
  // Can't handle non-static or non-unit strides
  if (!insertSliceOp || !(insertSliceOp.hasUnitStride()))
    return std::nullopt;
  auto insertSliceResult = insertSliceOp.getResult();
  if (!(insertSliceResult.hasOneUse()))
    return std::nullopt;
  if (auto dummyLayoutConversion = dyn_cast<UnrealizedConversionCastOp>(
          (*(insertSliceResult.use_begin())).getOwner()))
    return dummyLayoutConversion->getResult(0);
  return std::nullopt;
}

// Given a linalg::Conv2DNhwcFhwcOp, check that its inputs are computed
// by inverse pack ops, or slice of inverse pack ops. Optionally,
// 1. Check that its outputs are inserted into a full tensor that is
//    fed to an inverse unpack op (if checkOutput is true).
// 2. Check that its inputs are loop-invariant (if checkIfInputIsLoopInvariant
// is true).
bool isWrappedConv2DOp(linalg::Conv2DNhwcFhwcOp &conv2DOp, bool checkOutput,
                       bool checkIfInputIsLoopInvariant) {
  if (!isCandidate16BitElements(conv2DOp))
    return false;

  auto isApplicableInput =
      [&](std::optional<Value> dummyUnpackedInput) -> bool {
    if (!dummyUnpackedInput.has_value())
      return false;
    return (
        !checkIfInputIsLoopInvariant ||
        (dummyUnpackedInput.value().getParentBlock() != conv2DOp->getBlock()));
  };

  auto dummyUnpackInput = getPackedInput(conv2DOp, 0);
  if (!(isApplicableInput(dummyUnpackInput)))
    return false;

  auto dummyUnpackFilter = getPackedInput(conv2DOp, 1);
  if (!(isApplicableInput(dummyUnpackFilter)))
    return false;

  if (checkOutput) {
    auto dummyPackOutput = getPackedOutput(conv2DOp);
    return dummyPackOutput.has_value();
  }
  return true;
}

// Checks if an scf::ForOp's body has exactly 2 ops, and
// that its first op is another scf::ForOp, and that its result
// is produced by the child scf::ForOp
bool isParentLoop(scf::ForOp op) {
  auto &opsInBody = op.getBody()->getOperations();
  if (opsInBody.size() != 2)
    return false;
  if (auto childForOp = dyn_cast<scf::ForOp>(opsInBody.begin())) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(++(opsInBody.begin()))) {
      if ((yieldOp.getNumOperands() == 1) && (childForOp.getNumResults() == 1))
        return yieldOp.getOperand(0) == childForOp.getResult(0);
    }
  }
  return false;
}

// Helper to recursively delete an op and its dead operands
void eraseWithOperands(Operation *op, IRRewriter &rewriter) {
  // Collect operands before erasing the user
  SmallVector<Value> operands(op->getOperands());

  rewriter.eraseOp(op); // Erase the root op

  // Check operands
  for (Value operand : operands) {
    // If the operand is an OpResult (not a BlockArgument) and is now dead
    if (auto *defOp = operand.getDefiningOp()) {
      if (defOp->use_empty() && isOpTriviallyDead(defOp)) {
        eraseWithOperands(defOp, rewriter); // Recurse
      }
    }
  }
}

/// Select for loops containing conv2d operations operating on tensor slices,
/// and yielding the full tensor after insertion of the tiled result.
/// The conv2d ops should be wrapped by inverse_pack and inverse_unpack ops.
bool PreprocessTiledConv2DPass::populateCandidateLoops(
    scf::ForOp op, CandidateLoopsTy &candidates,
    ProcessedConvMap &processedConvOpsMap, IRRewriter &rewriter) {
  DBG("Number of results:\n" << op.getNumResults() << "\n");
  DBG("For Op:\n" << op << "\n");

  if (op.getNumResults() != 1)
    return false;

  Value forResult = op.getResult(0);
  DBG("forResult: " << forResult << "\nType: " << forResult.getType() << "\n");

  if (!isa<RankedTensorType>(forResult.getType()))
    return false;

  auto *terminator = op.getBody()->getTerminator();
  auto yieldOp = cast<mlir::scf::YieldOp>(terminator);

  if (isParentLoop(op)) {
    // This is a nested loop case. This is a candidate if inner loop
    // is a candidate
    auto innerForOp = cast<scf::ForOp>((op.getBody()->getOperations()).begin());
    return populateCandidateLoops(innerForOp, candidates, processedConvOpsMap,
                                  rewriter);
  }

  auto insertSliceOp =
      yieldOp.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSliceOp)
    return false;

  // Check that the insert slice's source is a linalg.conv2d op
  auto wrappedConv2DOp =
      insertSliceOp.getSource().getDefiningOp<linalg::Conv2DNhwcFhwcOp>();
  if (!wrappedConv2DOp)
    return false;
  // Don't consider this conv2d op for preprocessing again - it will
  // be processed as part of this loop, if it is a candidate.
  processedConvOpsMap.insert({wrappedConv2DOp, true});

  // Check that the conv2d op's inputs are slices of full tensors produced
  // by inverse pack ops. We already know that its output is inserted
  // into a full tensor.
  if (!isWrappedConv2DOp(wrappedConv2DOp, false, true))
    return false;

  DBG("selected candidate: " << op);
  loopToConvOp.insert({op, wrappedConv2DOp});
  return true;
}

Value PreprocessTiledConv2DPass::createAndInitTensor(IRRewriter &rewriter,
                                                     RankedTensorType RTT,
                                                     Location loc) {

  Value newOutTensor = tensor::EmptyOp::create(rewriter, loc, RTT.getShape(),
                                               RTT.getElementType());
  auto zeroF16Attr = rewriter.getF16FloatAttr(0.0);
  auto zeroF16 = arith::ConstantOp::create(rewriter, loc, zeroF16Attr);
  return (linalg::FillOp::create(rewriter, loc, ValueRange{zeroF16},
                                 ValueRange{newOutTensor}))
      .getResult(0);
}

Value PreprocessTiledConv2DPass::createPackedImageSlice(
    IRRewriter &rewriter, const llvm::DenseMap<Value, Value> &loopVarsMap,
    linalg::Conv2DNhwcFhwcOp convOp, Location &loc) {

  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<int64_t> sizes, strides;
  auto packedInput =
      getPackedInputWithSliceInfo(convOp, 0, offsets, sizes, strides);
  if (offsets.size()) // if this is a slice, create packed image slice
    return getPackedImageSlice(rewriter, packedInput, loc, offsets, sizes,
                               strides, loopVarsMap);
  return packedInput; // not a slice - return full packed input
}

Value PreprocessTiledConv2DPass::createPackedFilterSlice(
    IRRewriter &rewriter, const llvm::DenseMap<Value, Value> &loopVarsMap,
    linalg::Conv2DNhwcFhwcOp convOp, Location &loc) {

  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<int64_t> sizes, strides;
  auto packedInput =
      getPackedInputWithSliceInfo(convOp, 1, offsets, sizes, strides);
  if (offsets.size()) // if this is a slice, create packed filter slice
    return getPackedFilterSlice(rewriter, packedInput, loc, offsets, sizes,
                                strides, loopVarsMap);
  return packedInput; // not a slice - return full packed filter
}

// Given a linalg::Conv2DNhwcFhwcOp, populates the offsets, sizes and
// strides for the insert_slice op consuming the output of the conv2d op.
void getOutSlicePosition(linalg::Conv2DNhwcFhwcOp oldConvOp,
                         llvm::SmallVector<OpFoldResult> &offsets,
                         llvm::SmallVector<int64_t> &sizes) {
  // Find the insert slice op that inserts the output tile into full output
  // tensor.
  auto insertSliceOp = cast<tensor::InsertSliceOp>(
      (*(oldConvOp.getResult(0).use_begin())).getOwner());
  offsets = insertSliceOp.getMixedOffsets();
  auto staticSizes = insertSliceOp.getStaticSizes();
  sizes.assign(staticSizes.begin(), staticSizes.end());
}

// Insert the slice of packed output into the full packed output tensor.
// Given: unpacked slice info, map from variables used to compute unpacked
// slice info to variables to be used to compute packed slice info.
tensor::InsertSliceOp insertOutputSlice(
    IRRewriter &rewriter, Value packedFullOutput, Value packedSliceOutput,
    const llvm::SmallVector<OpFoldResult> &unpackedSliceOffsets,
    const llvm::SmallVector<int64_t> &unpackedSliceSizes,
    const llvm::DenseMap<Value, Value> &loopVarsMap, Location &loc) {
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
  // Static sizes with strides = 1. Only offsets are dynamic.
  for (int i = 0; i < 8; ++i) {
    packedSliceStrides.push_back(1);
    packedSliceOffsetsAsInt.push_back(ShapedType::kDynamic);
  }
  auto packedFullOutputRTT = cast<RankedTensorType>(packedFullOutput.getType());

  return mlir::tensor::InsertSliceOp::create(
      rewriter, loc, packedFullOutputRTT, packedSliceOutput, packedFullOutput,
      packedSliceOffsets, emptyVecVals, emptyVecVals, packedSliceOffsetsAsInt,
      packedSliceSizes, packedSliceStrides);
}

tensor::InsertSliceOp PreprocessTiledConv2DPass::transformWrappedConv2DOpHelper(
    IRRewriter &rewriter, linalg::Conv2DNhwcFhwcOp convOp,
    Value packedPartialOutput, const llvm::DenseMap<Value, Value> &loopVarsMap,
    Location &loc) {

  // Create packed image slice
  auto packedImageSlice =
      createPackedImageSlice(rewriter, loopVarsMap, convOp, loc);
  DBG("Packed Image Slice: " << packedImageSlice << "\n");

  // Create packed filter slice
  auto packedFilterSlice =
      createPackedFilterSlice(rewriter, loopVarsMap, convOp, loc);
  DBG("Packed Filter Slice: " << packedFilterSlice << "\n");

  // create inverse pack of image and filter
  auto unpackedImageSlice = convOp.getDpsInputOperand(0)->get();
  auto unpackedImageSliceRTT =
      cast<RankedTensorType>(unpackedImageSlice.getType());
  auto inversePackedImageSlice = UnrealizedConversionCastOp::create(
      rewriter, loc, mlir::TypeRange(unpackedImageSliceRTT),
      mlir::ValueRange({packedImageSlice}));

  auto unpackedFilterSlice = convOp.getDpsInputOperand(1)->get();
  auto unpackedFilterSliceRTT =
      cast<RankedTensorType>(unpackedFilterSlice.getType());
  auto inversePackedFilterSlice = UnrealizedConversionCastOp::create(
      rewriter, loc, mlir::TypeRange(unpackedFilterSliceRTT),
      mlir::ValueRange({packedFilterSlice}));

  // Init output of new conv2d op, consuming unpackedImageSlice and
  // unpackedFilterslice
  auto unpackedOutputSlice = convOp.getOutputs().front();
  auto unpackedOutputSliceRTT =
      cast<RankedTensorType>(unpackedOutputSlice.getType());
  auto initSliceOut =
      createAndInitTensor(rewriter, unpackedOutputSliceRTT, loc);

  // Create new conv2d op with unpackedImageSlice and unpackedFilterSlice
  // TODO: Copy attributes from old conv2d op to new conv2d op
  auto newConv2DSliceOp = linalg::Conv2DNhwcFhwcOp::create(
      rewriter, loc,
      mlir::ValueRange({inversePackedImageSlice.getResult(0),
                        inversePackedFilterSlice.getResult(0)}),
      mlir::ValueRange({initSliceOut}));

  // inverse unpack op on output of new conv2d op
  RankedTensorType inverseUnpackedSliceOutType =
      getPacked16BitElementType(unpackedOutputSliceRTT);
  auto inverseUnpackedSliceOut = UnrealizedConversionCastOp::create(
      rewriter, loc, mlir::TypeRange(inverseUnpackedSliceOutType),
      mlir::ValueRange({newConv2DSliceOp.getResult(0)}));

  // insert this slice into packed partial output

  // first get unpacked output slice offsets, sizes and strides
  llvm::SmallVector<OpFoldResult> outSliceOffsets;
  llvm::SmallVector<int64_t> outSliceSizes;
  getOutSlicePosition(convOp, outSliceOffsets, outSliceSizes);

  // insert packed output slice into full packed output - we can only
  // handle slices with static, unit strides
  auto insertSliceOp = insertOutputSlice(
      rewriter, packedPartialOutput, inverseUnpackedSliceOut.getResult(0),
      outSliceOffsets, outSliceSizes, loopVarsMap, loc);

  return insertSliceOp;
}

scf::ForOp PreprocessTiledConv2DPass::transformTiledConv2DLoop(
    CandidateLoopTy oldTiledLoop, IRRewriter &rewriter, Value initVal,
    const llvm::DenseMap<Value, Value> &parentMap) {
  if (!initVal) // Set insertion point only for outermost loop
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
  mlir::OperandRange initArgs = oldTiledLoop.getInitArgs();
  Value initPackedOut;

  // is the loop carried variable produced by inverse unpack op?
  if (initVal) {
    // an inner loop - use the initVal provided by the parent
    initPackedOut = initVal;
  } else {
    // create a new packed tensor and use it as the loop carried variable
    if (auto inversePackOp =
            (*(initArgs.begin())).getDefiningOp<UnrealizedConversionCastOp>()) {
      // Use the packed tensor as the initial value
      initPackedOut = inversePackOp.getInputs()[0];
    } else {
      // No initial value, create a new one initialized with 0's.
      initPackedOut = createAndInitTensor(rewriter, newResultRTT, loc);
    }
  }

  // Create a new tiled loop over the new packed tensor.
  auto newLoop =
      scf::ForOp::create(rewriter, loc, oldLB, oldUB, oldStep, initPackedOut);

  rewriter.setInsertionPointToStart(newLoop.getBody());
  llvm::DenseMap<Value, Value> loopVarsMap;
  if (initVal) {
    // This is not the outermost loop. We need to make sure
    // inner loops have access to outer loops' loop variables,
    // so pass them.
    // If initVal is nullptr, then this is the outermost loop,
    // and parentMap will be empty
    loopVarsMap.insert(parentMap.begin(), parentMap.end());
  }
  Value newCarried = newLoop.getRegionIterArgs().front();
  Value oldCarried = oldTiledLoop.getRegionIterArgs().front();
  loopVarsMap.insert({oldCarried, newCarried});
  loopVarsMap.insert(
      {oldTiledLoop.getInductionVar(), newLoop.getInductionVar()});
  Value newLoopResult;
  if (isParentLoop(oldTiledLoop)) {
    auto oldChildLoop =
        cast<scf::ForOp>(oldTiledLoop.getBody()->getOperations().front());
    auto newChildLoop = transformTiledConv2DLoop(oldChildLoop, rewriter,
                                                 newCarried, loopVarsMap);
    newLoopResult = newChildLoop.getResult(0);
  } else {
    auto it = loopToConvOp.find(oldTiledLoop);
    assert(it != loopToConvOp.end());
    linalg::Conv2DNhwcFhwcOp convOp =
        cast<linalg::Conv2DNhwcFhwcOp>(it->second);

    auto insertSliceOp = transformWrappedConv2DOpHelper(
        rewriter, convOp, newCarried, loopVarsMap, loc);

    newLoopResult = insertSliceOp.getResult();
  }
  scf::YieldOp::create(rewriter, loc, newLoopResult);
  rewriter.setInsertionPointAfter(newLoop);

  DBG("newTiledLoop: " << newLoop << "\n");
  DBG("Full Function:" << newLoop->getParentOfType<func::FuncOp>() << "\n");

  if (initVal) {
  } else {
    // This is the outermost loop.
    // The only use of old tiled loop's result is in a
    // builtin.unrealized_conversion_cast. Replace uses of result of that
    // builtin.unrealized_conversion_cast with the result of the new for loop.
    assert(oldLoopResult.hasOneUse());
    auto inverseUnpackOp = cast<UnrealizedConversionCastOp>(
        (*(oldTiledLoop.getResult(0).use_begin())).getOwner());

    // Replace uses of inverseUnrealizedCastOp with the result of new tiled
    // loop.
    inverseUnpackOp.getResult(0).replaceAllUsesWith(newLoop.getResult(0));
    rewriter.eraseOp(inverseUnpackOp);
  }
  return newLoop;
}

void PreprocessTiledConv2DPass::transformWrappedConv2DOp(
    linalg::Conv2DNhwcFhwcOp conv2DOp, IRRewriter &rewriter) {
  Location loc = conv2DOp.getLoc();
  auto oldResult = conv2DOp.getResult(0);
  auto oldResultRTT = mlir::cast<mlir::RankedTensorType>(oldResult.getType());
  auto newResultRTT = getPacked16BitElementType(oldResultRTT);
  auto insertSliceOp =
      dyn_cast<tensor::InsertSliceOp>((*(oldResult.use_begin())).getOwner());
  assert(insertSliceOp);
  Value packedPartialOutput;
  if (auto inversePackOp =
          insertSliceOp.getDest().getDefiningOp<UnrealizedConversionCastOp>()) {
    packedPartialOutput = inversePackOp.getInputs()[0];
  } else {
    packedPartialOutput = createAndInitTensor(rewriter, newResultRTT, loc);
  }
  llvm::DenseMap<Value, Value> emptyMap;
  auto newInsertSliceOp = transformWrappedConv2DOpHelper(
      rewriter, conv2DOp, packedPartialOutput, emptyMap, loc);
  auto inverseUnpackOp = cast<UnrealizedConversionCastOp>(
      (*(insertSliceOp.getResult().use_begin())).getOwner());
  inverseUnpackOp.getResult(0).replaceAllUsesWith(newInsertSliceOp.getResult());
  rewriter.eraseOp(inverseUnpackOp);
  assert(insertSliceOp->use_empty());
  rewriter.eraseOp(insertSliceOp);
}

void PreprocessTiledConv2DPass::runOnOperation() {

  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  CandidateLoopsTy candidateLoops;
  CandidateConvsTy candidateConvOps;
  ProcessedConvMap processedConvOpsMap;

  funcOp.walk([&](scf::ForOp op) {
    if (dyn_cast<func::FuncOp>(op->getParentOp()))
      if (populateCandidateLoops(op, candidateLoops, processedConvOpsMap,
                                 rewriter))
        candidateLoops.emplace_back(op);
    return WalkResult::advance();
  });

  funcOp.walk([&](linalg::Conv2DNhwcFhwcOp op) {
    if ((dyn_cast<func::FuncOp>(op->getParentOp()) &&
         (isWrappedConv2DOp(op, true, false)) &&
         processedConvOpsMap.find(op) == processedConvOpsMap.end())) {
      candidateConvOps.emplace_back(op);
    }
    return WalkResult::advance();
  });

  for (auto op : candidateLoops) {
    transformTiledConv2DLoop(op, rewriter);
    // cleanup
    eraseWithOperands(op, rewriter);
  }

  for (auto conv2DOp : candidateConvOps) {
    transformWrappedConv2DOp(conv2DOp, rewriter);
    // cleanup
    eraseWithOperands(conv2DOp, rewriter);
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createPreprocessTiledConv2DPass() {
  return std::make_unique<PreprocessTiledConv2DPass>();
}
