//===- TilingPass.cpp - tiling to enable vectorization in inner loop ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg op tiling so that inner most loop can be
// trivially vectorized by subsequent vectorization pass.
//
// TODO: We may need to investigate whether 'tiling for vectorization' could
//       be simplified by just applying interchange-vector to create innermost
//       loop range larger than target vec-length. Then instead of actually
//       tiling inner most loop (generating scf with steps), leave it as large
//       vectors that LLVM backend can optimize upon better. This may remove
//       the need for padding as well (for vectorization).
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/SplitReductionUtils.h"

#define DEBUG_TYPE "hexagon-tiling"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONTILING
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {
/// Given a linalgOp containing `nested loop with depth > 1`, returns an
/// interchange vector so that range of the innermost is least as large
/// as `minTileSize`. Returns false if constraints  are not satisfied or
/// a viable interchange vector is not found.
static bool computeInterchangeVector(linalg::LinalgOp op, unsigned minTileSize,
                                     std::vector<unsigned> &interchangeVector) {
  DBG("-> computing interchange vector for min-tile-size : " << minTileSize);

  for (auto iter : llvm::enumerate(op.getIteratorTypesArray())) {
    if (!linalg::isParallelIterator(iter.value())) {
      DBG(" -> interchange aborted. loops are not parallel.");
      return false;
    }
  }

  auto numLoops = op.getNumLoops();
  auto innermostLevel = numLoops - 1;
  auto ranges = op.getStaticLoopRanges();
  if (numLoops < 2 || llvm::any_of(ranges, [](uint64_t r) {
        return ShapedType::isDynamic(r);
      })) {
    DBG(" -> interchange aborted. dynamic shape or no nested loop.");
    return false;
  }
  assert(minTileSize > ranges[innermostLevel] &&
         "interchange vector requested when its not needed");

  assert(interchangeVector.size() == numLoops);
  std::iota(interchangeVector.begin(), interchangeVector.end(), 0);
  for (int i = innermostLevel - 1; i >= 0; --i) {
    if (ranges[i] >= minTileSize) {
      DBG("-> interchangeVector " << i << " with " << innermostLevel);
      std::swap(interchangeVector[innermostLevel], interchangeVector[i]);
      return true;
    }
  }
  return false;
}

/// Apply built-in linalg::tiling using `tiling options`.
static LogicalResult applyTiling(IRRewriter &rewriter, linalg::LinalgOp op,
                                 linalg::LinalgTilingOptions &options) {
  rewriter.setInsertionPoint(op);

  FailureOr<linalg::TiledLinalgOp> tiledOp =
      linalg::tileLinalgOp(rewriter, op, options);
  if (failed(tiledOp)) {
    return failure();
  }
  rewriter.replaceOp(op, tiledOp->tensorResults);
  return success();
}

static LogicalResult tileLinalgOp(linalg::LinalgOp op,
                                  bool useInterchangeVector,
                                  bool splitTilingRange) {
  // Early exit.
  unsigned numLoops = op.getNumLoops();
  if (!op.hasPureTensorSemantics() || !op->getNumResults() ||
      !isa<DestinationStyleOpInterface>(op.getOperation()) || numLoops < 1) {
    DBG("-> tiling aborted. constraints not satisfied");
    return failure();
  }

  linalg::LinalgOp tilingTarget = op;
  linalg::LinalgTilingOptions tileOption;
  SmallVector<int64_t, 10> tileSizes(numLoops, 1);
  bool needsInterchange = false;
  std::vector<unsigned> interchangeVector(numLoops, 0);
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(tilingTarget);

  auto dataTileSize = computeDataTileSize(op);
  if (!dataTileSize.has_value())
    return failure();

  tileSizes[numLoops - 1] = dataTileSize.value();

  auto ranges = op.getStaticLoopRanges();
  auto innerLoopRange = ranges[numLoops - 1];

  if (!ShapedType::isDynamic(innerLoopRange)) {
    if (innerLoopRange < dataTileSize) {
      needsInterchange = true;
      // Note that default is false as currently not sure
      // if trade-offs with transpose is right.
      if (!useInterchangeVector) {
        // Instead of considering this as a failure,  we might want
        // to flatten the tensor to make it tileable if possible
        DBG(" -> inner loop not tilable. Needs interchange");
        return failure();
      }
      if (!computeInterchangeVector(op, dataTileSize.value(),
                                    interchangeVector)) {
        return failure();
      }
      DBG(" -> inner loop dimension not compatible for vectorization");
      return failure();
    } else if (innerLoopRange == dataTileSize &&
               op.getNumParallelLoops() == 1) {
      DBG(" -> inner dim is already of data tile size, hence tiling for "
          "vectorization is not required");
      return failure();
    }

    if (needsInterchange) {
      tileOption.setInterchange(interchangeVector);
      SmallVector<int64_t, 10> newTileSizes(numLoops);
      for (int i = 0; i < numLoops; ++i)
        newTileSizes[i] = tileSizes[interchangeVector[i]];
      tileSizes = newTileSizes;
    }

    // If the innermost loop's range is not a multiple of tile size, we either
    // do padding or split the op into two parts - tileable part and remainder.
    if (!isPerfectlyTileable(innerLoopRange, dataTileSize.value())) {
      DBG("Innermost loop range (" << innerLoopRange << ") is not a multiple "
                                   << "of the tile-size ("
                                   << dataTileSize.value() << ")");
      // do padding
      if (!splitTilingRange) {
        DBG("Padding the inner-most loop");
        linalg::LinalgOp paddedOp;
        linalg::LinalgPaddingOptions paddingOption;
        SmallVector<Value> replacements;
        SmallVector<int64_t, 10> padDims;
        SmallVector<int64_t, 10> padToMultipleOf;
        SmallVector<tensor::PadOp> newPadOps;

        // Pad the dimension corresponding to the most inner loop
        // making it's range a multiple of tile size.
        padDims.push_back(numLoops - 1);
        padToMultipleOf.push_back(dataTileSize.value());

        llvm::ArrayRef<int64_t> padDimsRef(padDims.data(), padDims.size());
        paddingOption.setPaddingDimensions(padDimsRef);

        // The alternative option is to use 'materialize in destination'.
        // However, that can lead to RAW conflict error during bufferization
        // if two linalg ops (DPS) inits are same value (tensors) and both
        // lead to 'materialize in destination' directive after padding.
        paddingOption.copyBackOp =
            mlir::linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

        llvm::ArrayRef<int64_t> padToMultipleOfRef(padToMultipleOf.data(),
                                                   padToMultipleOf.size());
        paddingOption.setPadToMultipleOf(padToMultipleOfRef);

        DBG("Padding dimension  " << padDims[0] << " to become a multiple of "
                                  << padToMultipleOf[0]);
        // Tranform the op to a padded op. The transformation might fail if
        // the size and value of padding cannot be infered correctly from the
        // paddingOption, and also if the shape requested to be pad is not
        // static. Note: The latter should not be the reason for failure in our
        // code, as we have already ruled out that case before reaching to this
        // point.
        if (failed(linalg::rewriteAsPaddedOp(rewriter, op, paddingOption,
                                             paddedOp, replacements,
                                             newPadOps))) {
          return failure();
        }
        DBG("-->padded  Op :" << paddedOp);
        rewriter.replaceOp(tilingTarget, replacements);
        tilingTarget = paddedOp;

      } else {
        // Split the op into two parts,
        // accross the dimension corresponding to the most inner loop.
        // The firstPart contains the part that is mutiple of tilesize and the
        // secondPart is the remainder.
        assert(innerLoopRange > dataTileSize.value() &&
               "Splitting a range smaller than the tile size is requested");
        DBG("Splitting the innermost loop to seperate the tileable part from "
            "the remainder part");

        SmallVector<Operation *> firstPart, secondPart;
        unsigned splitDim = numLoops - 1;
        OpFoldResult splitpoint = rewriter.getIndexAttr(
            innerLoopRange - innerLoopRange % dataTileSize.value());
        DBG("Splitting at dimension  "
            << splitDim << " after point "
            << innerLoopRange % dataTileSize.value());

        std::tie(firstPart.emplace_back(), secondPart.emplace_back()) =
            linalg::splitOp(rewriter, cast<TilingInterface>(op.getOperation()),
                            splitDim, splitpoint);

        // At this point the original op has already been replaced by the
        // splitted parts. We just need to tile the first part.
        tilingTarget = cast<linalg::LinalgOp>(firstPart[0]);
      }
    }
  }

  tileOption.setTileSizes(tileSizes);
  return applyTiling(rewriter, tilingTarget, tileOption);
}

struct HexagonTilingPass : public ::impl::HexagonTilingBase<HexagonTilingPass> {
  explicit HexagonTilingPass(const HexagonTilingOptions &options)
      : HexagonTilingBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.walk([&](linalg::LinalgOp op) {
      bool appliedSplitReduction = false;
      IRRewriter rewriter(&getContext());
      if (enableSplitReduction && isa<linalg::GenericOp>(op) &&
          IsSplitReductionCandidate(op)) {
        if ((op.getNumLoops() >= 1) &&
            op.getIteratorTypesArray()[(op.getNumLoops() - 1)] ==
                utils::IteratorType::reduction) {
          // Perform split reduction if innermost dimension is reduction
          if (succeeded(SplitReductionLinalgOp(op))) {
            DBG("-> split reduction succeeded.");
            appliedSplitReduction = true;
          } else {
            DBG("-> split reduction failed.");
          }
        }
      }
      if (!appliedSplitReduction) {
        DBG("tiling candidate: " << op);
        if (succeeded(
                tileLinalgOp(op, useInterchangeVector, splitTilingRange))) {
          DBG("-> tiling succeeded.");
        } else {
          DBG("-> tiling failed.");
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonTilingPass(const HexagonTilingOptions &options) {
  return std::make_unique<HexagonTilingPass>(options);
}
