//===- SeedLayoutConversions.cpp ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Refer to SeedLayoutConversions pass description for details.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/PackUnpackUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "seed-layout-conversions"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SEEDLAYOUTCONVERSIONS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

using CandidateTy = linalg::Conv2DNhwcFhwcOp;
using CandidatesTy = SmallVector<CandidateTy>;
using CandidateQTy = linalg::Conv2DNhwcHwcfQOp;
using CandidatesQTy = SmallVector<CandidateQTy>;

struct SeedLayoutConversionsPass
    : public ::impl::SeedLayoutConversionsBase<SeedLayoutConversionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

// This helper function generates code for layout conversions for an FP16
// filter. It also preserves the original filter's type with a
// builtin.unrealized_conversion_cast op to avoid breaking the IR.
// Layout conversion for filters are expressed as a sequence of
// Pad, Reshape and Transpose ops.
mlir::UnrealizedConversionCastOp
packF16Conv2DFilter(IRRewriter &rewriter, Value filter,
                    RankedTensorType &filterRTTy,
                    mlir::arith::ConstantOp &padValue, Location &loc) {

  auto origFilterShape = filterRTTy.getShape();
  SmallVector<int64_t> paddingEnd = {(32 - (origFilterShape[0] % 32)) % 32, 0,
                                     0, (32 - (origFilterShape[3] % 32)) % 32};
  SmallVector<int64_t> paddingBeg = {0, 0, 0, 0};
  auto paddingEndAsOpFoldResult =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), paddingEnd);
  auto paddingBegAsOpFoldResult =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), paddingBeg);
  SmallVector<int64_t> paddedFilterShape = {
      origFilterShape[0] + paddingEnd[0], origFilterShape[1],
      origFilterShape[2], origFilterShape[3] + paddingEnd[3]};
  RankedTensorType paddedFilterRTTy =
      RankedTensorType::get(paddedFilterShape, filterRTTy.getElementType());

  auto paddedFilter = rewriter.create<tensor::PadOp>(
      loc, paddedFilterRTTy, filter, paddingBegAsOpFoldResult,
      paddingEndAsOpFoldResult, padValue, false);
  SmallVector<int64_t> reshapedPaddedFilterShape = {paddedFilterShape[0] / 32,
                                                    32,
                                                    paddedFilterShape[1],
                                                    paddedFilterShape[2],
                                                    paddedFilterShape[3] / 32,
                                                    16,
                                                    2};

  RankedTensorType reshapedPaddedFilterRTTy = RankedTensorType::get(
      reshapedPaddedFilterShape, filterRTTy.getElementType());
  SmallVector<Value> valueShape;
  for (int64_t val : reshapedPaddedFilterShape) {
    valueShape.push_back(
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(val)));
  }
  auto shapeAsTensor = rewriter.create<tensor::FromElementsOp>(loc, valueShape);
  auto reshapedPaddedFilter = rewriter.create<tensor::ReshapeOp>(
      loc, reshapedPaddedFilterRTTy, paddedFilter, shapeAsTensor);

  RankedTensorType packedFilterF16Type =
      getPackedFilter16BitElementType(filterRTTy);
  auto emptyPackedFilter =
      rewriter.create<tensor::EmptyOp>(loc, packedFilterF16Type.getShape(),
                                       packedFilterF16Type.getElementType());
  SmallVector<int64_t> permutation = {0, 4, 2, 3, 5, 1, 6};
  auto packedFilter = rewriter.create<linalg::TransposeOp>(
      loc, reshapedPaddedFilter, emptyPackedFilter, permutation);

  return rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, mlir::TypeRange(filterRTTy),
      mlir::ValueRange({packedFilter.getResult()}));
}

// Utility: Get RankedTensorType and shape
std::pair<RankedTensorType, ArrayRef<int64_t>> getRTTypeAndShape(Value val) {
  auto rtt = llvm::dyn_cast<RankedTensorType>(val.getType());
  assert(rtt && "Expecting RankedTensorType");
  assert(rtt.hasStaticShape());
  return {rtt, rtt.getShape()};
}

// Helper: Create pad value
arith::ConstantOp createPadValue(IRRewriter &rewriter, Location loc,
                                 Type elemType) {

  // Allow callers to pass shaped types by accident; normalize to element type.
  if (auto st = dyn_cast<ShapedType>(elemType))
    elemType = st.getElementType();

  // Quantized (per-tensor): use the zero-point from the quantized type.
  if (auto uq = dyn_cast<quant::UniformQuantizedType>(elemType)) {
    auto storageTy = cast<IntegerType>(uq.getStorageType());
    int64_t zp = uq.getZeroPoint();
    auto attr = IntegerAttr::get(storageTy, zp);
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }

  // Quantized (per-axis): all zero-points must be identical to form a scalar
  // pad.
  if (auto uqa = dyn_cast<quant::UniformQuantizedPerAxisType>(elemType)) {
    auto storageTy = cast<IntegerType>(uqa.getStorageType());
    auto zps = uqa.getZeroPoints();
    int64_t zp = zps.front();
    bool allEqual = llvm::all_of(zps, [zp](int64_t z) { return z == zp; });
    if (!allEqual) {
      llvm::report_fatal_error(
          "Per-axis quantization has differing zero-points; cannot form a "
          "single scalar pad value.");
    }
    auto attr = IntegerAttr::get(storageTy, zp);
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }

  // Non-quantized: keep prior behavior.
  // 0.0f for f16 and 0 for integers
  if (elemType.isF16())
    return rewriter.create<arith::ConstantOp>(loc,
                                              rewriter.getF16FloatAttr(0.0f));

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    auto attr = IntegerAttr::get(intTy, 0);
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }

  llvm_unreachable("Unsupported element type for pad value");
}

enum class Conv2DKind { F16, Q };

// ------------------------------
// F16 path: 2-stage pack + filter pack
// ------------------------------
void insertLayoutTransformsF16(Operation *conv2DOp, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(conv2DOp);
  auto loc = conv2DOp->getLoc();

  // Operands: image (NHWC), filter (FHWC)
  Value image = conv2DOp->getOperand(0);
  auto [imageRTTy, imageShape] = getRTTypeAndShape(image);

  Value filter = conv2DOp->getOperand(1);
  auto [filterRTTy, filterShape] = getRTTypeAndShape(filter);

  // Pad value (based on input element type)
  arith::ConstantOp padValue =
      createPadValue(rewriter, loc, imageRTTy.getElementType());

  // Packing parameters (F16):
  // - temp: pack NHWC using CROUTON shape (3 dims)
  SmallVector<int64_t> f16TempInnerDimsPos = {1, 2, 3}; // H, W, C
  SmallVector<int64_t> f16TempInnerTilesVals = {
      hexagon::F16_CROUTON_SHAPE[0],
      hexagon::F16_CROUTON_SHAPE[1] * hexagon::F16_CROUTON_SHAPE[3],
      hexagon::F16_CROUTON_SHAPE[2]};
  auto f16TempInnerTiles = mlir::getAsIndexOpFoldResult(rewriter.getContext(),
                                                        f16TempInnerTilesVals);

  // final: additional packing on dim 5 with tile 2
  SmallVector<int64_t> f16FinalInnerDimsPos = {5};
  SmallVector<int64_t> f16FinalInnerTilesVals = {2};
  auto f16FinalInnerTiles = mlir::getAsIndexOpFoldResult(
      rewriter.getContext(), f16FinalInnerTilesVals);

  // Pack input for F16 path: 2-stage
  auto newConv2DPackOp = buildF16PackOps(
      rewriter, image, imageRTTy, loc, padValue, f16TempInnerDimsPos,
      f16TempInnerTiles, f16FinalInnerDimsPos, f16FinalInnerTiles);
  UnrealizedConversionCastOp newConv2DInput =
      rewriter.create<mlir::UnrealizedConversionCastOp>(
          loc, mlir::TypeRange(imageRTTy),
          mlir::ValueRange({newConv2DPackOp.getResult()}));
  // Pack filter for F16 path
  Value newConv2DFilter =
      packF16Conv2DFilter(rewriter, filter, filterRTTy, padValue, loc)
          .getResult(0);

  // Output type/shape and empty tensor
  Value out = conv2DOp->getResult(0);
  auto [outRTTy, outShape] = getRTTypeAndShape(out);
  Value emptyConv2DOut =
      rewriter.create<tensor::EmptyOp>(loc, outShape, outRTTy.getElementType());

  // New Conv2D op (F16)
  Operation *newConv2DOp = rewriter.create<linalg::Conv2DNhwcFhwcOp>(
      loc, mlir::ValueRange({newConv2DInput.getResult(0), newConv2DFilter}),
      mlir::ValueRange({emptyConv2DOut}));

  // Unpack output back to original layout
  // First, insert unrealized conversion cast to packed type.
  auto firstUnpackInput =
      rewriter
          .create<mlir::UnrealizedConversionCastOp>(
              loc, mlir::TypeRange(getPacked16BitElementType(outRTTy)),
              mlir::ValueRange({newConv2DOp->getResult(0)}))
          .getResult(0);
  linalg::UnPackOp unpackedOrigResult = unpackF16Output(
      rewriter, firstUnpackInput, loc,
      /*final dims*/ f16FinalInnerDimsPos, f16FinalInnerTiles,
      /*temp dims*/ f16TempInnerDimsPos, f16TempInnerTiles, outRTTy, outShape);

  // Replace old Conv2D result
  conv2DOp->getResult(0).replaceAllUsesWith(unpackedOrigResult.getResult());
  rewriter.eraseOp(conv2DOp);
}

// ------------------------------
// I8 quantized path: 1-stage pack, filter passthrough
// ------------------------------
void insertLayoutTransformsI8Q(Operation *conv2DOp, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(conv2DOp);
  auto loc = conv2DOp->getLoc();

  // Gather qpath cast chain (scast -> dcast -> qcast) after Conv
  Operation *scastOp = *conv2DOp->getResult(0).getUsers().begin();
  assert(scastOp && "Expected scastOp to be non-null");
  Operation *dcastOp = *scastOp->getResult(0).getUsers().begin();
  assert(dcastOp && "Expected dcastOp to be non-null");
  Operation *qcastOp = *dcastOp->getResult(0).getUsers().begin();
  assert(qcastOp && "Expected qcastOp to be non-null");

  // Operands: image (NHWC), filter (HWCF), zp's
  Value image = conv2DOp->getOperand(0);
  auto [imageRTTy, imageShape] = getRTTypeAndShape(image);

  Value filter = conv2DOp->getOperand(1);
  Value act_zp = conv2DOp->getOperand(2);
  Value wt_zp = conv2DOp->getOperand(3);

  // Pad value
  arith::ConstantOp padValue =
      createPadValue(rewriter, loc, imageRTTy.getElementType());

  // Packing parameters (I8): single-stage pack over [H, W, C]
  SmallVector<int64_t> i8InnerDimsPos = {1, 2, 3}; // H, W, C for NHWC
  SmallVector<int64_t> i8InnerTilesVals = {hexagon::INT8_CROUTON_SHAPE[0],
                                           hexagon::INT8_CROUTON_SHAPE[1],
                                           hexagon::INT8_CROUTON_SHAPE[2]};
  auto i8InnerTiles =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), i8InnerTilesVals);

  // Pack input for I8 path
  mlir::linalg::PackOp newConv2DPackOp = buildI8PackOps(
      rewriter, image, imageRTTy, loc, padValue, i8InnerDimsPos, i8InnerTiles);
  UnrealizedConversionCastOp newConv2DInput =
      rewriter.create<mlir::UnrealizedConversionCastOp>(
          loc, mlir::TypeRange(imageRTTy),
          mlir::ValueRange({newConv2DPackOp.getResult()}));

  // Filter remains as-is for I8 path
  Value newConv2DFilter = filter;

  // Output and empty
  Value out = conv2DOp->getResult(0);
  auto [outRTTy, outShape] = getRTTypeAndShape(out);
  Value emptyConv2DOut =
      rewriter.create<tensor::EmptyOp>(loc, outShape, outRTTy.getElementType());

  // New Conv2D op (quantized)
  Operation *newConv2DOp = rewriter.create<linalg::Conv2DNhwcHwcfQOp>(
      loc,
      mlir::ValueRange(
          {newConv2DInput.getResult(0), newConv2DFilter, act_zp, wt_zp}),
      mlir::ValueRange({emptyConv2DOut}));

  // Unpack occurs after the qcast op to match existing pipeline
  rewriter.setInsertionPointAfter(qcastOp);
  auto qcastOut = qcastOp->getResult(0);
  auto [qRTTy, qShape] = getRTTypeAndShape(qcastOut);

  auto unpackInput =
      rewriter
          .create<mlir::UnrealizedConversionCastOp>(
              loc, mlir::TypeRange(getPacked8BitElementType(qRTTy)),
              mlir::ValueRange({qcastOp->getResult(0)}))
          .getResult(0);

  linalg::UnPackOp unpacked = unpackI8Output(
      rewriter, unpackInput, loc, i8InnerDimsPos, i8InnerTiles, qRTTy, qShape);

  // Replace all uses except the one inside the cast created by unpack
  Value newVal = unpacked.getResult();
  Operation *castOp = newVal.getDefiningOp()->getOperand(0).getDefiningOp();
  qcastOut.replaceAllUsesExcept(newVal, castOp);

  // Wire the new conv result to where the old conv wrote
  conv2DOp->getResult(0).replaceAllUsesWith(newConv2DOp->getResult(0));
  rewriter.eraseOp(conv2DOp);
}

template <typename CandidateT>
void transformLayout(CandidateT op, IRRewriter &rewriter) {
  TypeSwitch<Operation *, void>(op.getOperation())
      .template Case<linalg::Conv2DNhwcFhwcOp>([&](auto conv2DOp) {
        insertLayoutTransformsF16(conv2DOp.getOperation(), rewriter);
      })
      .template Case<linalg::Conv2DNhwcHwcfQOp>([&](auto conv2DOp) {
        insertLayoutTransformsI8Q(conv2DOp.getOperation(), rewriter);
      })
      .Default([](auto) { llvm_unreachable("Not expecting a non-conv2D op"); });
}

// Utility: Collect candidate ops of a given type
template <typename ConvOpT, typename CandidateFn>
SmallVector<ConvOpT> collectCandidateOps(FunctionOpInterface funcOp,
                                         CandidateFn isCandidate) {
  SmallVector<ConvOpT> result;
  funcOp.walk([&](ConvOpT linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        isCandidate(linalgOp)) {
      result.push_back(linalgOp);
    }
  });
  return result;
}

// Helper to find a user op by name and validate single use
Operation *findSingleUseOp(Value val, StringRef opName) {
  // Check if there is exactly one user
  if (!val.hasOneUse())
    return nullptr;

  Operation *user = *val.getUsers().begin();
  return (user->getName().getStringRef() == opName) ? user : nullptr;
}

void SeedLayoutConversionsPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Helper: Check quantization chain (scast → dcast → qcast)
  auto hasQuantChain = [](Operation *convOp) -> bool {
    auto scastOp = findSingleUseOp(convOp->getResult(0), "quant.scast");
    if (!scastOp)
      return false;

    auto dcastOp = findSingleUseOp(scastOp->getResult(0), "quant.dcast");
    if (!dcastOp)
      return false;

    auto qcastOp = findSingleUseOp(dcastOp->getResult(0), "quant.qcast");
    return qcastOp != nullptr;
  };

  // Collect candidates for F16 type
  auto conv2DOps = collectCandidateOps<linalg::Conv2DNhwcFhwcOp>(
      funcOp, isCandidate16BitElements);

  // Collect candidates for I8 type with quantization chain
  auto Qconv2DOps = collectCandidateOps<linalg::Conv2DNhwcHwcfQOp>(
      funcOp, [&](linalg::Conv2DNhwcHwcfQOp op) {
        return isCandidate8BitElements(op) && hasQuantChain(op.getOperation());
      });

  // Utility: Transform all ops in a container
  auto transformAll = [&](auto &ops) {
    for (auto linalgOp : ops) {
      transformLayout(linalgOp, rewriter);
    }
  };

  // Transform both sets
  transformAll(conv2DOps);
  transformAll(Qconv2DOps);
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createSeedLayoutConversionsPass() {
  return std::make_unique<SeedLayoutConversionsPass>();
}
