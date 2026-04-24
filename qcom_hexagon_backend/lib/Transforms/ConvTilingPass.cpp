//===- ConvTilingPass.cpp - tile the linalg convolution ops       ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Transforms a regular convolution op into a tiled convolution op.
// Currently only linalg::Conv2DNhwcFhwcOp is considered for tiling.
//
// Convolution tiling is supported for crouton and non-crouton shapes.
// The non-crouton shapes undergoes the following sequence of transformations
// post tiling:
// 1. Run canonicalizer to eliminate single iteration loops so that the
// outermost loop over N is eliminated. This makes the amenable for loop
// peeling.
// 2. Peel the last iteration of the tiled conv2D loop. This pass generates
// 2 loop nests. The first loop nest contains crouton shapes and the second
// loop nest has non-crouton shape.
// 3. Run canonicalizer again so that the following transformations happen:
//    a. Affine::min and Affine::apply are eliminated.
//    b. The loop for the last peeled iteration is also simplified and
//    eliminated.
// 4. Add pack-unpack op pair around the conv2D op. This pattern is needed
// for later passes.
//
// Consider the following example containing a non-crouton tensor shape:
// %5 = builtin.unrealized_conversion_cast %transposed :
// tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x128xf16>
// %7 = linalg.conv_2d_nhwc_fhwc
// ins(%3, %5 : tensor<1x239x27x128xf16>, tensor<64x1x1x128xf16>)
// outs(%6 : tensor<1x239x27x64xf16>) -> tensor<1x239x27x64xf16>
// %8 = builtin.unrealized_conversion_cast %7 :
// tensor<1x239x27x64xf16> to tensor<1x30x7x2x8x2x32x2xf16>
//
// The generated tiled loop is shown below:
// %6 = scf.for %arg2 = %c0 to %c239 step %c32 iter_args(%arg3 = %5) ->
// (tensor<1x239x27x64xf16>) {
//   %10 = affine.min #map(%arg2)
//   %11 = affine.min #map(%arg2)
//
//   %extracted_slice = tensor.extract_slice
//   %2[0, %arg2, 0, 0] [1, %10, 27, 128] [1, 1, 1, 1] :
//   tensor<1x239x27x128xf16> to tensor<1x?x27x128xf16>
//
//   %extracted_slice_3 = tensor.extract_slice
//   %arg3[0, %arg2, 0, 0] [1, %11, 27, 64] [1, 1, 1, 1] :
//   tensor<1x239x27x64xf16> to tensor<1x?x27x64xf16>
//
//   %12 = linalg.conv_2d_nhwc_fhwc
//   ins(%extracted_slice, %4 : tensor<1x?x27x128xf16>, tensor<64x1x1x128xf16>)
//   outs(%extracted_slice_3 : tensor<1x?x27x64xf16>) -> tensor<1x?x27x64xf16>
//
//   %inserted_slice = tensor.insert_slice %12 into
//   %arg3[0, %arg2, 0, 0] [1, %11, 27, 64] [1, 1, 1, 1] :
//   tensor<1x?x27x64xf16> into tensor<1x239x27x64xf16>
//
//   scf.yield %inserted_slice : tensor<1x239x27x64xf16>
// }
//
// The tiled conv2d op goes through canonicalization and loop peeling.
// The tiled and peeled op is shown below:
// L1: %4 = builtin.unrealized_conversion_cast %transposed :
// tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x128xf16>
// %6 = scf.for %arg2 = %c0 to %c224 step %c32 iter_args(%arg3 = %5)
// -> (tensor<1x239x27x64xf16>) {
//    %extracted_slice_4 = tensor.extract_slice
//    %2[0, %arg2, 0, 0] [1, 32, 27, 128] [1, 1, 1, 1] :
//    tensor<1x239x27x128xf16> to tensor<1x32x27x128xf16>
//
//    %extracted_slice_5 = tensor.extract_slice
//    %arg3[0, %arg2, 0, 0] [1, 32, 27, 64] [1, 1, 1, 1] :
//    tensor<1x239x27x64xf16> to tensor<1x32x27x64xf16>
//
//    %13 = linalg.conv_2d_nhwc_fhwc
//    ins(%extracted_slice_4, %4 : tensor<1x32x27x128xf16>,
//    tensor<64x1x1x128xf16>) outs(%extracted_slice_5 : tensor<1x32x27x64xf16>)
//    -> tensor<1x32x27x64xf16>
//
//    %inserted_slice_6 = tensor.insert_slice %13 into
//    %arg3[0, %arg2, 0, 0] [1, 32, 27, 64] [1, 1, 1, 1] :
//    tensor<1x32x27x64xf16> into tensor<1x239x27x64xf16>
//
//    scf.yield %inserted_slice_6 : tensor<1x239x27x64xf16>
// }
//
// %extracted_slice = tensor.extract_slice
// %2[0, 224, 0, 0] [1, 15, 27, 128] [1, 1, 1, 1] :
// tensor<1x239x27x128xf16> to tensor<1x15x27x128xf16>
//
// %extracted_slice_2 = tensor.extract_slice
// %6[0, 224, 0, 0] [1, 15, 27, 64] [1, 1, 1, 1] :
// tensor<1x239x27x64xf16> to tensor<1x15x27x64xf16>
//
// %9 = linalg.conv_2d_nhwc_fhwc
// ins(%extracted_slice, %4 : tensor<1x15x27x128xf16>, tensor<64x1x1x128xf16>)
// outs(%extracted_slice_2 : tensor<1x15x27x64xf16>) -> tensor<1x15x27x64xf16>
//
// %inserted_slice = tensor.insert_slice %9 into
// %6[0, 224, 0, 0] [1, 15, 27, 64] [1, 1, 1, 1] :
// tensor<1x15x27x64xf16> into tensor<1x239x27x64xf16>
//
// L2: %10 = builtin.unrealized_conversion_cast %inserted_slice :
// tensor<1x239x27x64xf16> to tensor<1x30x7x2x8x2x32x2xf16>
//
// The tiled and peeled conv2d op is subsequently wrapped in conversion casts
// before and after an convolution op. When the convolution op is tiled and
// peeled, the conversion casts do not exist between the main loop and the last
// peeled iteration. HMX lowering pass expects the conv2d in this pattern.
// Please refer to the slide 9 of the above presentation:
//
// Consider the above code that contains tiled and peeled conv2D op:
//
// The tiled and peeled loop has two conv_2d ops between the
// unrealized_conversion_casts L1 and L2. Unrealized_conversion_casts T1 and T2
// are added between the two conv2d_ops as follows:
//
// %4 = builtin.unrealized_conversion_cast %transposed :
// tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x128xf16>
// %6 = scf.for %arg2 = %c0 to %c224 step %c32 iter_args(%arg3 = %5)
// -> (tensor<1x239x27x64xf16>) {
//    %extracted_slice_4 = tensor.extract_slice
//    %2[0, %arg2, 0, 0] [1, 32, 27, 128] [1, 1, 1, 1] :
//    tensor<1x239x27x128xf16> to tensor<1x32x27x128xf16>
//
//    %extracted_slice_5 = tensor.extract_slice
//    %arg3[0, %arg2, 0, 0] [1, 32, 27, 64] [1, 1, 1, 1] :
//    tensor<1x239x27x64xf16> to tensor<1x32x27x64xf16>
//
//    %13 = linalg.conv_2d_nhwc_fhwc
//    ins(%extracted_slice_4, %4 : tensor<1x32x27x128xf16>,
//    tensor<64x1x1x128xf16>) outs(%extracted_slice_5 : tensor<1x32x27x64xf16>)
//    -> tensor<1x32x27x64xf16>
//
//    %inserted_slice_6 = tensor.insert_slice %13 into
//    %arg3[0, %arg2, 0, 0] [1, 32, 27, 64] [1, 1, 1, 1] :
//    tensor<1x32x27x64xf16> into tensor<1x239x27x64xf16>
//
//    scf.yield %inserted_slice_6 : tensor<1x239x27x64xf16>
//  }
//
//  T1: %7 = builtin.unrealized_conversion_cast %6 :
//  tensor<1x239x27x64xf16> to tensor<1x30x7x2x8x2x32x2xf16>
//
//  T2: %8 = builtin.unrealized_conversion_cast %7 :
//  tensor<1x30x7x2x8x2x32x2xf16> to tensor<1x239x27x64xf16>
//
//  %extracted_slice = tensor.extract_slice
//  %2[0, 224, 0, 0] [1, 15, 27, 128] [1, 1, 1, 1] :
//  tensor<1x239x27x128xf16> to tensor<1x15x27x128xf16>
//
//  %extracted_slice_2 = tensor.extract_slice
//  %8[0, 224, 0, 0] [1, 15, 27, 64] [1, 1, 1, 1] :
//  tensor<1x239x27x64xf16> to tensor<1x15x27x64xf16>
//
//  %9 = linalg.conv_2d_nhwc_fhwc
//  ins(%extracted_slice, %4 : tensor<1x15x27x128xf16>, tensor<64x1x1x128xf16>)
//  outs(%extracted_slice_2 : tensor<1x15x27x64xf16>) -> tensor<1x15x27x64xf16>
//
//  %inserted_slice = tensor.insert_slice %9 into
//  %8[0, 224, 0, 0] [1, 15, 27, 64] [1, 1, 1, 1] :
//  tensor<1x15x27x64xf16> into tensor<1x239x27x64xf16>
//
//  %10 = builtin.unrealized_conversion_cast %inserted_slice :
//  tensor<1x239x27x64xf16> to tensor<1x30x7x2x8x2x32x2xf16>
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/PackUnpackUtils.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "-conv-tiling"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_CONVTILING
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct Conv2DTilingPattern final
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
  Conv2DTilingPattern(MLIRContext *context,
                      const DenseMap<int64_t, int64_t> &tileSizes)
      : OpRewritePattern(context), userProvidedTileSizes(tileSizes) {}

  // Find linalg::Conv2DNhwcFhwcOp and tile the op, wherever possible
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp,
                                PatternRewriter &) const override;

private:
  const DenseMap<int64_t, int64_t> &userProvidedTileSizes;
};

// Function that tiles the input convOp.
// We want to tile only along Height and Output Channels (OC) dimensions.
// For all other dimensions, the tile size is set to the loop iteration range.
// Currently, tiling is supported only along one dimension.
LogicalResult
Conv2DTilingPattern::matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                     PatternRewriter &rewriter) const {
  auto isCandidateConvOp = [](linalg::Conv2DNhwcFhwcOp op) -> bool {
    if (!isa<func::FuncOp>(op->getParentOp()) ||
        !isCandidate16BitElements(op)) {
      return false;
    }

    return true;
  };

  if (!isCandidateConvOp(convOp)) {
    return rewriter.notifyMatchFailure(convOp, " not a candidate conv op.\n");
  }

  // Get the loop iteration ranges.
  SmallVector<int64_t> tileSizes = getInitialTileSize(convOp);
  LLVM_DEBUG(llvm::dbgs() << "LOOP BOUNDS:"; for (int64_t t
                                                  : tileSizes) llvm::dbgs()
                                             << t << " ";);

  // Record the tiling preferences in the object `tilingOptions'
  linalg::LinalgTilingOptions tilingOptions;
  tilingOptions.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

  // Use the tilingFactor provided by the user
  if (!userProvidedTileSizes.empty()) {
    // Height must be a multiple of 8
    if (userProvidedTileSizes.find(1) != userProvidedTileSizes.end()) {
      int64_t heightTileSize = userProvidedTileSizes.lookup(1);
      if (heightTileSize % 8 != 0) {
        return rewriter.notifyMatchFailure(
            convOp, " Tiling factor must be a multiple of 8 for " +
                        std::to_string(userProvidedTileSizes.lookup(1)));
      } else {
        tileSizes[1] = heightTileSize;
      }
    }

    // OC must be a multiple of 32
    if (userProvidedTileSizes.find(3) != userProvidedTileSizes.end()) {
      int64_t ocTileSize = userProvidedTileSizes.lookup(3);
      if (ocTileSize % 32 != 0) {
        return rewriter.notifyMatchFailure(
            convOp, " Tiling factor must be a multiple of 32 for " +
                        std::to_string(userProvidedTileSizes.lookup(3)));
      } else {
        tileSizes[3] = ocTileSize;
      }
    }
  } else {
    // the user has not provided a tilingFactor
    // Get the best tiling factor for the dim
    SmallVector<int64_t> tilingDims = {1, 3};
    // Compute the best tile size for the user provided tilingDim
    std::optional<SmallVector<int64_t>> bestTileSizeForTilingDim =
        determineTileSizes(convOp, /*vtcmBudget=*/0, tilingDims);

    // Return if the best tile size is not computed
    if (!bestTileSizeForTilingDim) {
      return rewriter.notifyMatchFailure(
          convOp, "Could not get the best tile size for ");
    }

    // Set the tile size to the computed tile size
    tileSizes[1] = (*bestTileSizeForTilingDim)[1];
    tileSizes[3] = (*bestTileSizeForTilingDim)[3];
    // Print the best tile size for the tiling dim
    DBG("The computed tile size is dimension 1"
        << tileSizes[1] << " and for dimension 3 " << tileSizes[3]);
  }

  // Set the tile sizes
  tilingOptions.setTileSizes(tileSizes);

  LLVM_DEBUG(llvm::dbgs() << "TILE SIZES:\n"; for (int64_t t
                                                   : tileSizes) llvm::dbgs()
                                              << t << " ";
             llvm::dbgs() << "END\n");

  // Perform the tiling. This will generate the scf.for loop,
  // tensor.extract_slice, and tensor.insert_slice operations.
  // The result contains the new loop and the tiled operation.
  // Note: The `tileLinalgOp` utility expects `tileSizes` to correspond to
  // the dimensions of the *iteration space*, not necessarily the operand.
  // For linalg ops, the iteration space often directly maps to output
  // dimensions. So, if output is NHWC, C_out is the 4th dimension (index 3).
  // Set tile size for dimension 3. Other dimensions will not be tiled.
  auto tiledResults = linalg::tileLinalgOp(rewriter, convOp, tilingOptions);
  if (failed(tiledResults)) {
    return rewriter.notifyMatchFailure(convOp, "failed to tile linalg.conv2d");
  }

  // Replace the original operation with the results of the tiling.
  rewriter.replaceOp(convOp, tiledResults->tensorResults);
  return success();
}

// Function that generate unrealized_conversion_cast ops
// between the full crouton conv2d and the partial peeled conv2d op.
//
// Any redundant pack/unpack ops will be eliminated during subsequent
// canonicalizations.
static LogicalResult genPackUnpackOpPair(func::FuncOp funcOp) {
  // Process only those tiled conv2D op whose input tensor is generated by an
  // UnrealizedConversionCastOp (unpack op)
  auto getProducerCastOp =
      [](linalg::Conv2DNhwcFhwcOp *candConvOp) -> UnrealizedConversionCastOp {
    // Get operands
    Operation *op0 = candConvOp->getOperand(0).getDefiningOp();
    Operation *op1 = candConvOp->getOperand(1).getDefiningOp();

    // Input tensor or the filter for a tiled conv2D ops is generated through an
    // ExtractSliceOp
    tensor::ExtractSliceOp extractOp0 =
        dyn_cast_or_null<tensor::ExtractSliceOp>(op0);
    tensor::ExtractSliceOp extractOp1 =
        dyn_cast_or_null<tensor::ExtractSliceOp>(op1);

    // Consider any of the inputs based on the following conditions:
    // 1. Input tensor and filter both are tiled: Ensure both extract slice ops
    // are in the same for loop
    // 2. Only input tensor is tiled: Consider input tensor
    // 3. Only filter is tiled: Consider filter
    // 4. Input tensor and filter are not tiled: return.
    tensor::ExtractSliceOp extractOp;
    if (!extractOp0) {
      // Case 4: The conv2D is not tiled
      if (!extractOp1)
        return nullptr;
      // Case 3
      extractOp = extractOp1;
    } else {
      // Case 1
      if (extractOp1) {
        if (extractOp0->getParentOp() != extractOp1->getParentOp())
          return nullptr;
      }
      // Case 2
      extractOp = extractOp0;
    }

    // Get the full tensor
    Operation *inputOp = extractOp.getSource().getDefiningOp();

    return dyn_cast_or_null<UnrealizedConversionCastOp>(inputOp);
  };

  // Check if the forOp is the outermost tiled forOp for a conv2DOp
  auto isCandidateForOp = [](scf::ForOp forOp) -> bool {
    if (!isa<func::FuncOp>(forOp->getParentOp()))
      return false;

    if (forOp.getNumResults() != 1)
      return false;

    Value forResult = forOp.getResult(0);
    if (!isa<RankedTensorType>(forResult.getType()))
      return false;

    return true;
  };

  auto getInsertSliceOp = [](scf::ForOp forOp) -> tensor::InsertSliceOp {
    scf::ForOp parentForOp = forOp;
    while (auto childForOp = dyn_cast_or_null<scf::ForOp>(
               (parentForOp.getBody()->getOperations()).begin())) {
      parentForOp = childForOp;
    }

    auto *terminator = parentForOp.getBody()->getTerminator();
    auto yieldOp = cast<mlir::scf::YieldOp>(terminator);
    if (!yieldOp)
      return nullptr;

    return yieldOp.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  };

  // Walk through the funcOp to generate unrealized_conversion_cast ops
  funcOp.walk([&](scf::ForOp forOp) {
    DBG("Processing Op:");
    LLVM_DEBUG(forOp.print(llvm::dbgs()));
    DBG("\n");

    if (!isCandidateForOp(forOp))
      return WalkResult::skip();

    auto insertSliceOp = getInsertSliceOp(forOp);
    if (!insertSliceOp)
      return WalkResult::skip();

    // Check that the insert slice's source is a linalg.conv2d op
    auto convOp =
        insertSliceOp.getSource().getDefiningOp<linalg::Conv2DNhwcFhwcOp>();
    if (!convOp)
      return WalkResult::skip();

    // Get the producer Unrealized Conversion Cast Op
    UnrealizedConversionCastOp producerCastOp = getProducerCastOp(&convOp);
    if (!producerCastOp)
      return WalkResult::skip();

    DBG("Found tiled Conv2D Op for inserting cast ops\n");

    IRRewriter rewriter(convOp.getContext());
    rewriter.setInsertionPointAfter(forOp);

    // Generate pack op
    auto resOp = forOp->getResult(0);
    auto resOpType = resOp.getType();
    auto packedType =
        getPacked16BitElementType(llvm::dyn_cast<RankedTensorType>(resOpType));
    auto packOp = mlir::UnrealizedConversionCastOp::create(
        rewriter, forOp->getLoc(), mlir::TypeRange(packedType),
        mlir::ValueRange({resOp}));

    // Generate unpack op
    auto unpackOp = mlir::UnrealizedConversionCastOp::create(
        rewriter, forOp->getLoc(), mlir::TypeRange(resOpType),
        mlir::ValueRange({packOp->getResult(0)}));

    // Replace only those uses of outermostForOp dominated by it with unpackOp
    mlir::DominanceInfo dom(funcOp);
    forOp->replaceUsesWithIf(unpackOp, [&](mlir::OpOperand &use) {
      mlir::Operation *user = use.getOwner();
      if (user->getBlock() == unpackOp->getBlock())
        return unpackOp->isBeforeInBlock(user);
      return dom.dominates(unpackOp, user);
    });

    return WalkResult::advance();
  });

  return success();
}

// Definition of ConvTilingPass
struct ConvTilingPass : public ::impl::ConvTilingBase<ConvTilingPass> {
  explicit ConvTilingPass(const ConvTilingOptions &options)
      : ConvTilingBase(options) {}

  void runOnOperation() override {
    std::optional<DenseMap<int64_t, int64_t>> userProvidedTileSizes =
        parseConvTileSizes(convTileSizes);

    if (!userProvidedTileSizes.has_value()) {
      DBG("Failed to parse the malformed input tile size string.\n");
      signalPassFailure();
      return;
    }

    //  Apply Conv2DTilingPattern on the funcOp
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<Conv2DTilingPattern>(patterns.getContext(),
                                      userProvidedTileSizes.value());
    // Walk and apply Conv2DTilingPattern on funcOp
    walkAndApplyPatterns(funcOp, std::move(patterns));

    // Simplify and peel the tiled conv2d.
    // Crouton aligned tensor shapes are not peeled in this pipeline.
    mlir::PassManager pm(funcOp->getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCFForLoopPeeling());
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(funcOp))) {
      DBG("Something failed in the post pass pipeline\n");
      signalPassFailure();
      return;
    }

    // Generate Inverse Pack Unpack Pair for easy HMX lowering
    if (mlir::failed(genPackUnpackOpPair(funcOp))) {
      DBG("Something failed during the generation of pack unpack operations\n");
    }
    return;
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createConvTilingPass(const ConvTilingOptions &options) {
  return std::make_unique<ConvTilingPass>(options);
}
