//===- FoldPackUnpackConstantsPass.cpp - Fold ops on constants ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass folds tensor operations on constant tensors at compile time:
//   - pack/unpack operations
//   - transpose operations
//   - expand_shape/collapse_shape operations
//   - pad operations
//
// This eliminates runtime overhead for transforming constant weights.
//
// Example:
//   %cst = arith.constant dense<...> : tensor<768xf16>
//   %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [32]
//             into %dest : tensor<768xf16> -> tensor<24x32xf16>
//
// Becomes:
//   %packed = arith.constant dense<...> : tensor<24x32xf16>
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {

using namespace mlir;

// Helper to create a constant (may be resource-based for large tensors, inline
// for small)
static Value createConstant(PatternRewriter &rewriter, Location loc,
                            RankedTensorType type,
                            DenseElementsAttr denseAttr) {
  // For large constants, create a resource-based constant to keep IR readable
  // Threshold: if the constant has more than 64 elements, use resource format
  constexpr int64_t kResourceThreshold = 64;

  if (type.getNumElements() > kResourceThreshold) {
    // Get the raw data from the dense attribute
    ArrayRef<char> rawData = denseAttr.getRawData();

    // Create an AsmResourceBlob from the raw data
    auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(rawData);

    // Create a resource attribute using the blob
    // MLIR's resource manager will append suffixes (_0, _1, etc.) to ensure
    // uniqueness
    auto resourceAttr = DenseResourceElementsAttr::get(type, "folded_constant",
                                                       std::move(blob));

    return arith::ConstantOp::create(rewriter, loc, type, resourceAttr);
  }

  // For small constants, use inline dense format
  return arith::ConstantOp::create(rewriter, loc, type, denseAttr);
}

// Helper to check if a value is a constant tensor and get its dense
// representation
static DenseElementsAttr getConstantTensor(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto attr = constOp.getValue();

    // Handle inline dense constants
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      return denseAttr;
    }

    // Handle resource-based dense constants.
    if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
      auto type = cast<ShapedType>(value.getType());

      // Fast path: if the resource has been materialized in the context,
      // try to read it via typed resource attrs.
      auto fromTypedResource =
          llvm::TypeSwitch<Attribute, DenseElementsAttr>(resourceAttr)
              .Case<DenseF32ResourceElementsAttr, DenseF64ResourceElementsAttr,
                    DenseI8ResourceElementsAttr, DenseUI8ResourceElementsAttr,
                    DenseI16ResourceElementsAttr, DenseUI16ResourceElementsAttr,
                    DenseI32ResourceElementsAttr, DenseUI32ResourceElementsAttr,
                    DenseI64ResourceElementsAttr,
                    DenseUI64ResourceElementsAttr>(
                  [&](auto typed) -> DenseElementsAttr {
                    if (auto arrayRef = typed.tryGetAsArrayRef())
                      return DenseElementsAttr::get(type, *arrayRef);
                    return nullptr;
                  })
              .Default([](Attribute) -> DenseElementsAttr { return nullptr; });
      if (fromTypedResource)
        return fromTypedResource;

      // Fallback: Try to access raw blob data directly. This works when the
      // resource is defined in the MLIR file.
      auto handle = resourceAttr.getRawHandle();
      if (auto *blob = handle.getBlob()) {
        if (blob->getData().data()) {
          // Note: f16 does not currently have a typed resource attr in MLIR, so
          // it typically relies on this raw-blob fallback.
          return DenseElementsAttr::getFromRawBuffer(type, blob->getData());
        }
      }

      // Resource could not be materialized - this can happen in test files
      // where resources aren't properly loaded into the MLIR context.
      return nullptr;
    }
  }
  return nullptr;
}

// Helper to get constant integer value from OpFoldResult
static std::optional<int64_t> getConstantInt(OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

// Helper to compute packed indices from unpacked indices
static SmallVector<int64_t> computePackedIndices(
    ArrayRef<int64_t> unpackedIndices, ArrayRef<int64_t> unpackedShape,
    ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> innerTileSizes,
    ArrayRef<int64_t> outerDimsPerm) {

  int64_t rank = unpackedShape.size();
  int64_t numInnerDims = innerDimsPos.size();

  // Start with outer indices
  SmallVector<int64_t> packedIndices;

  // Compute outer tile indices
  for (int64_t i = 0; i < rank; ++i) {
    int64_t idx = unpackedIndices[i];

    // Check if this dimension is tiled
    auto it = llvm::find(innerDimsPos, i);
    if (it != innerDimsPos.end()) {
      int64_t tileIdx = std::distance(innerDimsPos.begin(), it);
      int64_t tileSize = innerTileSizes[tileIdx];
      packedIndices.push_back(idx / tileSize);
    } else {
      packedIndices.push_back(idx);
    }
  }

  // Apply outer dims permutation if present
  if (!outerDimsPerm.empty()) {
    assert(outerDimsPerm.size() == static_cast<size_t>(rank) &&
           "outerDimsPerm must permute all outer dims");
    SmallVector<int64_t> permutedIndices(packedIndices.size());
    for (size_t i = 0; i < outerDimsPerm.size(); ++i) {
      permutedIndices[i] = packedIndices[outerDimsPerm[i]];
    }
    packedIndices = permutedIndices;
  }

  // Add inner tile indices
  for (int64_t i = 0; i < numInnerDims; ++i) {
    int64_t dimPos = innerDimsPos[i];
    int64_t tileSize = innerTileSizes[i];
    int64_t idx = unpackedIndices[dimPos];
    packedIndices.push_back(idx % tileSize);
  }

  return packedIndices;
}

// Helper to compute linear index from multi-dimensional indices
static int64_t computeLinearIndex(ArrayRef<int64_t> indices,
                                  ArrayRef<int64_t> shape) {
  int64_t linearIdx = 0;
  int64_t stride = 1;
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    linearIdx += indices[i] * stride;
    stride *= shape[i];
  }
  return linearIdx;
}

// Helper to compute multi-dimensional indices from linear index
static SmallVector<int64_t> computeMultiDimIndices(int64_t linearIdx,
                                                   ArrayRef<int64_t> shape) {
  SmallVector<int64_t> indices(shape.size());
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    indices[i] = linearIdx % shape[i];
    linearIdx /= shape[i];
  }
  return indices;
}

// Pattern to fold pack operations on constant tensors
struct FoldPackConstant : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(packOp.getSource());
    if (!sourceConst)
      return failure();

    // Only handle static shapes for now
    auto sourceType = cast<RankedTensorType>(packOp.getSource().getType());
    auto destType = cast<RankedTensorType>(packOp.getDestType());

    if (!sourceType.hasStaticShape() || !destType.hasStaticShape())
      return failure();

    // Get pack parameters
    ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
    // Only support static inner tile sizes.
    SmallVector<int64_t> innerTileSizes = packOp.getStaticTiles();
    if (llvm::any_of(innerTileSizes, ShapedType::isDynamic))
      return failure();
    if (llvm::any_of(innerTileSizes, [](int64_t s) { return s <= 0; }))
      return failure();
    ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty() &&
        outerDimsPerm.size() != static_cast<size_t>(sourceType.getRank()))
      return failure();

    // Get padding value if present
    Value paddingValue = packOp.getPaddingValue();
    Attribute paddingAttr;
    if (paddingValue) {
      if (auto padConst = paddingValue.getDefiningOp<arith::ConstantOp>()) {
        paddingAttr = padConst.getValue();
        auto typedAttr = dyn_cast<TypedAttr>(paddingAttr);
        if (!typedAttr || typedAttr.getType() != destType.getElementType())
          return failure();
      } else {
        return failure();
      }
    }

    // Compute the packed constant
    auto sourceShape = sourceType.getShape();
    auto destShape = destType.getShape();
    int64_t rank = sourceType.getRank();

    // Precompute mapping from unpacked dim -> tile index (or -1 if untiled).
    SmallVector<int64_t> dimToTile(rank, -1);
    for (auto [tileIdx, dimPos] : llvm::enumerate(innerDimsPos))
      dimToTile[dimPos] = static_cast<int64_t>(tileIdx);

    // Create a buffer for the packed data
    int64_t numElements = destType.getNumElements();
    SmallVector<Attribute> packedValues(numElements);

    // Iterate over all elements in the destination (packed) tensor
    for (int64_t i = 0; i < numElements; ++i) {
      auto packedIndices = computeMultiDimIndices(i, destShape);

      // Compute corresponding unpacked indices
      SmallVector<int64_t> unpackedIndices(sourceShape.size());

      // Extract outer tile indices (before permutation)
      SmallVector<int64_t> outerIndices;
      if (!outerDimsPerm.empty()) {
        // Reverse the permutation
        SmallVector<int64_t> invPerm(outerDimsPerm.size());
        for (size_t j = 0; j < outerDimsPerm.size(); ++j) {
          invPerm[outerDimsPerm[j]] = j;
        }
        for (size_t j = 0; j < sourceShape.size(); ++j) {
          outerIndices.push_back(packedIndices[invPerm[j]]);
        }
      } else {
        for (size_t j = 0; j < sourceShape.size(); ++j) {
          outerIndices.push_back(packedIndices[j]);
        }
      }

      // Compute unpacked indices
      for (size_t j = 0; j < sourceShape.size(); ++j) {
        int64_t tileIdx = dimToTile[j];
        if (tileIdx >= 0) {
          int64_t tileSize = innerTileSizes[tileIdx];
          int64_t innerIdx = packedIndices[sourceShape.size() + tileIdx];
          unpackedIndices[j] = outerIndices[j] * tileSize + innerIdx;
        } else {
          unpackedIndices[j] = outerIndices[j];
        }
      }

      // Check if this is a padded element
      bool isPadded =
          llvm::any_of(llvm::seq<size_t>(0, sourceShape.size()), [&](size_t j) {
            return unpackedIndices[j] >= sourceShape[j];
          });

      if (isPadded && !paddingAttr)
        return failure();

      packedValues[i] =
          isPadded ? paddingAttr
                   : sourceConst.getValues<Attribute>()[computeLinearIndex(
                         unpackedIndices, sourceShape)];
    }

    // Create the packed constant (may be resource-based for large tensors).
    auto packedAttr = DenseElementsAttr::get(destType, packedValues);
    auto packedConst =
        createConstant(rewriter, packOp.getLoc(), destType, packedAttr);

    rewriter.replaceOp(packOp, packedConst);

    return success();
  }
};

// Pattern to fold unpack operations on constant tensors
struct FoldUnpackConstant : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(unpackOp.getSource());
    if (!sourceConst)
      return failure();

    // Only handle static shapes for now
    auto sourceType = cast<RankedTensorType>(unpackOp.getSource().getType());
    auto destType = cast<RankedTensorType>(unpackOp.getDestType());

    if (!sourceType.hasStaticShape() || !destType.hasStaticShape())
      return failure();

    // Get unpack parameters
    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    // Only support static inner tile sizes.
    SmallVector<int64_t> innerTileSizes = unpackOp.getStaticTiles();
    if (llvm::any_of(innerTileSizes, ShapedType::isDynamic))
      return failure();
    if (llvm::any_of(innerTileSizes, [](int64_t s) { return s <= 0; }))
      return failure();
    ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty() &&
        outerDimsPerm.size() != static_cast<size_t>(destType.getRank()))
      return failure();

    // Compute the unpacked constant
    auto sourceShape = sourceType.getShape();
    auto destShape = destType.getShape();

    // Create a buffer for the unpacked data
    int64_t numElements = destType.getNumElements();
    SmallVector<Attribute> unpackedValues(numElements);

    // Iterate over all elements in the destination (unpacked) tensor
    for (int64_t i = 0; i < numElements; ++i) {
      auto unpackedIndices = computeMultiDimIndices(i, destShape);

      // Compute corresponding packed indices
      auto packedIndices =
          computePackedIndices(unpackedIndices, destShape, innerDimsPos,
                               innerTileSizes, outerDimsPerm);

      // Get value from source constant
      int64_t sourceLinearIdx = computeLinearIndex(packedIndices, sourceShape);
      if (sourceLinearIdx < 0 || sourceLinearIdx >= sourceType.getNumElements())
        return failure();
      unpackedValues[i] = sourceConst.getValues<Attribute>()[sourceLinearIdx];
    }

    // Create the unpacked constant (may be resource-based for large tensors).
    auto unpackedAttr = DenseElementsAttr::get(destType, unpackedValues);
    auto unpackedConst =
        createConstant(rewriter, unpackOp.getLoc(), destType, unpackedAttr);

    rewriter.replaceOp(unpackOp, unpackedConst);

    return success();
  }
};

// Pattern to fold transpose operations on constant tensors
struct FoldTransposeConstant : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // linalg.transpose is a destination-passing style op
    // Get the input operand (first operand in DPS inputs)
    Value input = transposeOp.getDpsInputOperand(0)->get();

    // Check if the input is a constant
    auto inputConst = getConstantTensor(input);
    if (!inputConst)
      return failure();

    // Only handle static shapes
    auto inputType = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(transposeOp.getResultTypes()[0]);

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return failure();

    // Get permutation
    ArrayRef<int64_t> permutation = transposeOp.getPermutation();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    // Validate permutation.
    int64_t rank = inputType.getRank();
    if (permutation.size() != static_cast<size_t>(rank))
      return failure();
    llvm::SmallBitVector seen(rank);
    for (int64_t p : permutation) {
      if (p < 0 || p >= rank || seen.test(p))
        return failure();
      seen.set(p);
    }

    // Create buffer for transposed data
    int64_t numElements = outputType.getNumElements();
    SmallVector<Attribute> transposedValues(numElements);

    // Iterate over all elements in the output tensor
    SmallVector<int64_t> inputIndices(inputShape.size());
    for (int64_t i = 0; i < numElements; ++i) {
      auto outputIndices = computeMultiDimIndices(i, outputShape);

      // Compute corresponding input indices using inverse permutation
      for (size_t j = 0; j < permutation.size(); ++j) {
        inputIndices[permutation[j]] = outputIndices[j];
      }

      // Get value from input constant
      int64_t inputLinearIdx = computeLinearIndex(inputIndices, inputShape);
      if (inputLinearIdx < 0 || inputLinearIdx >= inputType.getNumElements())
        return failure();
      transposedValues[i] = inputConst.getValues<Attribute>()[inputLinearIdx];
    }

    // Create the transposed constant (may be resource-based for large tensors).
    auto transposedAttr = DenseElementsAttr::get(outputType, transposedValues);
    auto transposedConst = createConstant(rewriter, transposeOp.getLoc(),
                                          outputType, transposedAttr);

    rewriter.replaceOp(transposeOp, transposedConst);

    return success();
  }
};

// Pattern to fold expand_shape operations on constant tensors
struct FoldExpandShapeConstant
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(expandOp.getSrc());
    if (!sourceConst)
      return failure();

    // Only handle static shapes
    auto sourceType = cast<RankedTensorType>(expandOp.getSrc().getType());
    auto resultType = cast<RankedTensorType>(expandOp.getType());

    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    if (sourceType.getNumElements() != resultType.getNumElements())
      return failure();

    // expand_shape is just a reshape - the data layout doesn't change
    // We can directly create a new constant with the expanded shape
    auto expandedAttr = sourceConst.reshape(resultType);
    auto expandedConst =
        createConstant(rewriter, expandOp.getLoc(), resultType, expandedAttr);

    rewriter.replaceOp(expandOp, expandedConst);

    return success();
  }
};

// Pattern to fold collapse_shape operations on constant tensors
struct FoldCollapseShapeConstant
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(collapseOp.getSrc());
    if (!sourceConst)
      return failure();

    // Only handle static shapes
    auto sourceType = cast<RankedTensorType>(collapseOp.getSrc().getType());
    auto resultType = cast<RankedTensorType>(collapseOp.getType());

    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    if (sourceType.getNumElements() != resultType.getNumElements())
      return failure();

    // collapse_shape is just a reshape - the data layout doesn't change
    // We can directly create a new constant with the collapsed shape
    auto collapsedAttr = sourceConst.reshape(resultType);
    auto collapsedConst = createConstant(rewriter, collapseOp.getLoc(),
                                         resultType, collapsedAttr);

    rewriter.replaceOp(collapseOp, collapsedConst);

    return success();
  }
};

// Pattern to fold pad operations on constant tensors
struct FoldPadConstant : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(padOp.getSource());
    if (!sourceConst)
      return failure();

    // Only handle static shapes and constant padding values
    auto sourceType = cast<RankedTensorType>(padOp.getSource().getType());
    auto resultType = cast<RankedTensorType>(padOp.getResultType());

    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    // Check if we have a constant padding value
    Value paddingValue = padOp.getConstantPaddingValue();
    if (!paddingValue)
      return failure();

    // Get the padding attribute
    auto paddingConst = paddingValue.getDefiningOp<arith::ConstantOp>();
    if (!paddingConst)
      return failure();

    Attribute paddingAttr = paddingConst.getValue();
    auto typedPaddingAttr = dyn_cast<TypedAttr>(paddingAttr);
    if (!typedPaddingAttr ||
        typedPaddingAttr.getType() != resultType.getElementType())
      return failure();

    // Get static low and high padding
    SmallVector<int64_t> lowPad, highPad;
    for (auto low : padOp.getStaticLow()) {
      if (low == ShapedType::kDynamic)
        return failure();
      lowPad.push_back(low);
    }
    for (auto high : padOp.getStaticHigh()) {
      if (high == ShapedType::kDynamic)
        return failure();
      highPad.push_back(high);
    }

    auto sourceShape = sourceType.getShape();
    auto resultShape = resultType.getShape();

    // Create buffer for padded data
    int64_t numElements = resultType.getNumElements();
    SmallVector<Attribute> paddedValues(numElements);

    // Iterate over all elements in the result tensor
    SmallVector<int64_t> sourceIndices(sourceShape.size());
    for (int64_t i = 0; i < numElements; ++i) {
      auto resultIndices = computeMultiDimIndices(i, resultShape);

      // Check if this element is in the padded region
      bool isPadded = false;
      for (size_t j = 0; j < resultIndices.size(); ++j) {
        int64_t idx = resultIndices[j] - lowPad[j];
        if (idx < 0 || idx >= sourceShape[j]) {
          isPadded = true;
          break;
        }
        sourceIndices[j] = idx;
      }

      if (isPadded) {
        paddedValues[i] = paddingAttr;
      } else {
        int64_t sourceLinearIdx =
            computeLinearIndex(sourceIndices, sourceShape);
        if (sourceLinearIdx < 0 ||
            sourceLinearIdx >= sourceType.getNumElements())
          return failure();
        paddedValues[i] = sourceConst.getValues<Attribute>()[sourceLinearIdx];
      }
    }

    // Create the padded constant (may be resource-based for large tensors).
    auto paddedAttr = DenseElementsAttr::get(resultType, paddedValues);
    auto paddedConst =
        createConstant(rewriter, padOp.getLoc(), resultType, paddedAttr);

    rewriter.replaceOp(padOp, paddedConst);

    return success();
  }
};

// Pattern to fold reshape operations on constant tensors
struct FoldReshapeConstant : public OpRewritePattern<tensor::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a constant
    auto sourceConst = getConstantTensor(reshapeOp.getSource());
    if (!sourceConst)
      return failure();

    // Only handle static shapes
    auto sourceType = cast<RankedTensorType>(reshapeOp.getSource().getType());
    auto resultType = cast<RankedTensorType>(reshapeOp.getType());

    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    // Check that the number of elements matches
    if (sourceType.getNumElements() != resultType.getNumElements())
      return failure();

    // reshape is just a reshape - the data layout doesn't change
    // We can directly create a new constant with the reshaped shape
    auto reshapedAttr = sourceConst.reshape(resultType);
    auto reshapedConst =
        createConstant(rewriter, reshapeOp.getLoc(), resultType, reshapedAttr);

    rewriter.replaceOp(reshapeOp, reshapedConst);

    return success();
  }
};

// Pass definition
struct FoldPackUnpackConstantsPass
    : public PassWrapper<FoldPackUnpackConstantsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldPackUnpackConstantsPass)

  void runOnOperation() override {
    auto module = getOperation();

    // Apply patterns to fold pack/unpack/transpose/reshape/pad operations on
    // constants
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldPackConstant, FoldUnpackConstant, FoldTransposeConstant,
                 FoldExpandShapeConstant, FoldCollapseShapeConstant,
                 FoldPadConstant, FoldReshapeConstant>(&getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "fold-pack-unpack-constants"; }

  StringRef getDescription() const final {
    return "Fold pack/unpack/transpose/reshape/expand_shape/collapse_shape/pad "
           "operations on constant tensors";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createFoldPackUnpackConstantsPass() {
  return std::make_unique<FoldPackUnpackConstantsPass>();
}

} // namespace hexagon
} // namespace mlir
