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
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Transforms/Passes.h"
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
  Conv2DTilingPattern(MLIRContext *context) : OpRewritePattern(context) {}

  // Find linalg::Conv2DNhwcFhwcOp and tile the op, wherever possible
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp,
                                PatternRewriter &) const override;

  // User provided tilingFactor
  static int64_t tilingFactor;
  // User provided tilingDim
  static unsigned tilingDim;
};

// Initialize the static variables
int64_t Conv2DTilingPattern::tilingFactor = -1;
unsigned Conv2DTilingPattern::tilingDim = 1;

// Function that tiles the input convOp.
// We want to tile only along Height and Output Channels (OC) dimensions.
// For all other dimensions, the tile size is set to the loop iteration range.
// Currently, tiling is supported only along one dimension.
LogicalResult
Conv2DTilingPattern::matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                     PatternRewriter &rewriter) const {
  // Get the loop iteration ranges.
  SmallVector<int64_t> tileSizes = getInitialTileSize(convOp);
  LLVM_DEBUG(llvm::dbgs() << "LOOP BOUNDS:"; for (int64_t t
                                                  : tileSizes) llvm::dbgs()
                                             << t << " ";);

  // Record the tiling preferences in the object `tilingOptions'
  linalg::LinalgTilingOptions tilingOptions;
  tilingOptions.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

  // Use the tilingFactor provided by the user
  if (tilingFactor != -1) {
    if (Conv2DTilingPattern::tilingDim == 1) {
      // Height must be a multiple of 8
      if (tilingFactor % 8 != 0)
        return rewriter.notifyMatchFailure(
            convOp, " Tiling factor must be a multiple of 8 for " +
                        std::to_string(Conv2DTilingPattern::tilingDim));
    } else if (Conv2DTilingPattern::tilingDim == 3) {
      // OC must be a multiple of 32
      if (tilingFactor % 32 != 0)
        return rewriter.notifyMatchFailure(
            convOp, " Tiling factor must be a multiple of 32 for " +
                        std::to_string(Conv2DTilingPattern::tilingDim));
    }

    tileSizes[Conv2DTilingPattern::tilingDim] = tilingFactor;
  } else {
    // the user has not provided a tilingFactor
    // Get the best tiling factor for the dim
    SmallVector<int64_t> tilingDims;
    tilingDims.push_back(Conv2DTilingPattern::tilingDim);
    // Compute the best tile size for the user provided tilingDim
    std::optional<SmallVector<int64_t>> bestTileSizeForTilingDim =
        determineTileSizes(convOp, tilingDims);

    // Return if the best tile size is not computed
    if (!bestTileSizeForTilingDim) {
      return rewriter.notifyMatchFailure(
          convOp, "Could not get the best tile size for " +
                      std::to_string(Conv2DTilingPattern::tilingDim));
    }

    // Set the tile size to the computed tile size
    tileSizes[Conv2DTilingPattern::tilingDim] =
        (*bestTileSizeForTilingDim)[Conv2DTilingPattern::tilingDim];

    // Print the best tile size for the tiling dim
    DBG("The computed tile size is "
        << tileSizes[Conv2DTilingPattern::tilingDim]);
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

// Definition of ConvTilingPass
struct ConvTilingPass : public ::impl::ConvTilingBase<ConvTilingPass> {
  explicit ConvTilingPass(const ConvTilingOptions &options)
      : ConvTilingBase(options) {}

  // Set the tilingDim
  void setTilingDim() {
    if (convTileOCDim) {
      // Tile OC dim
      Conv2DTilingPattern::tilingDim = 3;
      DBG("Tile factor of OC Dim:" << convTilingFactor);
    } else { // Tile height dim by default
      // Tile Height dim
      Conv2DTilingPattern::tilingDim = 1;
      DBG("Tile factor of Height Dim:" << convTilingFactor);
    }

    return;
  }

  void runOnOperation() override {
    // Set the tilingFactor
    Conv2DTilingPattern::tilingFactor = convTilingFactor;
    // Set the tilingDim
    setTilingDim();

    // Apply Conv2DTilingPattern on the funcOp
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<Conv2DTilingPattern>(patterns.getContext());

    // Walk and apply Conv2DTilingPattern on funcOp
    walkAndApplyPatterns(funcOp, std::move(patterns));

    return;
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createConvTilingPass(const ConvTilingOptions &options) {
  return std::make_unique<ConvTilingPass>(options);
}
