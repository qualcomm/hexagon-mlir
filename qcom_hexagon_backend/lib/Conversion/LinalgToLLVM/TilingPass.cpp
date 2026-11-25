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
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(tilingTarget);

  auto dataTileSize = computeDataTileSize(op);
  if (!dataTileSize.has_value()) {
    DBG("-> tiling aborted. Could not compute dataTileSize");
    return failure();
  }

  auto ranges = op.getStaticLoopRanges();
  auto innerLoopRange = ranges[numLoops - 1];

  // Currently tiling dynamic loops with default tiling, to follow
  // what we were doing. Uncomment this early bailing-out if we want
  // to change it and to never tile dynamic loops:
  // // Bail-out if the innermost loop has a dynamic range
  // if (ShapedType::isDynamic(innerLoopRange)) {
  //   DBG(" -> Tiling pass can't deal with dynamic loop range");
  //   return failure();
  // }

  // Bail-out if there's only 1 loop and the inner loop range fits
  // in a single HVX vector, since that requires no tiling, and we
  // want this to reach the vectorization pass as is, which will
  // vectorize at its natural width, to rely on the LLVM backend
  if (op.getNumLoops() == 1 && innerLoopRange <= dataTileSize.value()) {
    DBG(" -> Only 1 loop that fits in an HVX vector, tiling not required");
    return failure();
  }

  // Bail-out if there's only 1 loop, if it's a reduction, and if it fits in
  // a pair of HVX vectors, since the LLVM backend can handle it
  // Question: should we remove the "is a reduction" condition and also apply it
  // to elementwise op?
  if (op.getNumLoops() == 1 && innerLoopRange <= 2 * dataTileSize.value() &&
      op.getIteratorTypesArray()[0] == utils::IteratorType::reduction) {
    // Avoid tiling vector->scalar reductions when the inner loop dim is
    // at most 2x HVX vector length. Tiling would produce multiple
    // vector->scalar reductions, but the LLVM Hexagon backend can
    // lower this more efficiently as a valign to combine two vectors
    // followed by a single vector->scalar reduction (valign + vadd/vmax).
    DBG(" -> reduction to scalar fits in HVX reduction tree, skip tiling");
    return failure();
  }

  // Default tile sizes are [1, 1, ..., dataTileSize]
  linalg::LinalgTilingOptions tileOption;
  SmallVector<int64_t, 10> tileSizes(numLoops, 1);
  tileSizes[numLoops - 1] = dataTileSize.value();

  // --- Perfect tiling scenario, that uses the default tile sizes ---
  // which also applies to dynamic loops to match the old behavor
  // (TODO: check that we really want to tile dynamic loops)
  if (isPerfectlyTileable(innerLoopRange, dataTileSize.value()) ||
      ShapedType::isDynamic(innerLoopRange)) {
    tileOption.setTileSizes(tileSizes);
    return applyTiling(rewriter, tilingTarget, tileOption);
  }

  // At this point, we know it's not a perfect tiling scenario (loop's range
  // is not a multiple of tile size), so we'll attempt various tiling strategies
  DBG("Imperfect tiling: Innermost loop range ("
      << innerLoopRange << ") is not a multiple "
      << "of the tile-size (" << dataTileSize.value() << ")");

  // --- Strategy 1) Interchange ---
  // At this point, if (innerLoopRange < dataTileSize) holds, it means we have
  // more than one loop (otherwise we would have aborted earlier). So it makes
  // sense to try the interchange strategy
  if (innerLoopRange < dataTileSize.value()) {
    std::vector<unsigned> interchangeVector(numLoops, 0);
    // Note that default is false as currently not sure
    // if trade-offs with transpose is right.
    if (!useInterchangeVector) {
      // Instead of considering this as a failure,  we might want
      // to flatten the tensor to make it tileable if possible
      DBG(" -> inner loop would need interchange for being tilable");
      return failure();
    }
    // Attempt the interchange strategy
    bool interchangeResult =
        computeInterchangeVector(op, dataTileSize.value(), interchangeVector);
    if (!interchangeResult) {
      DBG(" -> Computing the interchange vector failed.");
      return failure();
    }

    // Apply the interchangeVector obtained
    tileOption.setInterchange(interchangeVector);
    SmallVector<int64_t, 10> newTileSizes(numLoops);
    for (int i = 0; i < numLoops; ++i)
      newTileSizes[i] = tileSizes[interchangeVector[i]];
    tileSizes = newTileSizes;

    tileOption.setTileSizes(tileSizes);
    return applyTiling(rewriter, tilingTarget, tileOption);
  } // End of Interchange strategy

  // --- Strategy 2: Padding ---
  // If splitTilingRange is not set, we will try a naive padding that pads with
  // the value 0, as long as it's not a reduction op (otherwise this naive
  // padding could change the semantics of the op)
  if (!splitTilingRange && op.getIteratorTypesArray()[numLoops - 1] !=
                               utils::IteratorType::reduction) {
    DBG("Padding the inner-most loop");
    linalg::LinalgOp paddedOp;
    linalg::LinalgPaddingOptions paddingOption;
    SmallVector<Value> replacements;
    SmallVector<int64_t, 10> padDims;
    SmallVector<int64_t, 10> padToMultipleOf;
    SmallVector<tensor::PadOp> newPadOps;

    // Pad the dimension corresponding to the most inner loop
    // making its range a multiple of tile size.
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
    // paddingOption
    if (failed(linalg::rewriteAsPaddedOp(rewriter, op, paddingOption, paddedOp,
                                         replacements, newPadOps))) {
      DBG("Padding failed");
      return failure();
    }
    DBG("-->padded  Op :" << paddedOp);
    rewriter.replaceOp(tilingTarget, replacements);
    tilingTarget = paddedOp;

    tileOption.setTileSizes(tileSizes);
    return applyTiling(rewriter, tilingTarget, tileOption);
  }

  // --- Strategy 3: Splitting ---
  // Split across the dimension corresponding to the most inner loop.
  // The firstPart contains the part that is multiple of tilesize and the
  // secondPart is the remainder.
  if (splitTilingRange) {
    // assert should hold because < dataTileSize was routed to
    // interchange and == to perfect tiling, so only > non-multiples reach here.
    assert(innerLoopRange > dataTileSize.value() &&
           "Splitting a range smaller than the tile size is requested");
    DBG("Splitting the innermost loop to separate the tileable part from "
        "the remainder part");

    SmallVector<Operation *> firstPart, secondPart;
    unsigned splitDim = numLoops - 1;
    OpFoldResult splitpoint = rewriter.getIndexAttr(
        innerLoopRange - innerLoopRange % dataTileSize.value());
    DBG("Splitting at dimension  " << splitDim << " after point "
                                   << innerLoopRange % dataTileSize.value());

    std::tie(firstPart.emplace_back(), secondPart.emplace_back()) =
        linalg::splitOp(rewriter, cast<TilingInterface>(op.getOperation()),
                        splitDim, splitpoint);

    // At this point the original op has already been replaced by the
    // splitted parts. We just need to tile the first part.
    tilingTarget = cast<linalg::LinalgOp>(firstPart[0]);

    tileOption.setTileSizes(tileSizes);
    return applyTiling(rewriter, tilingTarget, tileOption);
  }

  // If we reach this point, we have exhausted the tiling strategies to try
  DBG("Could not find any tiling strategy to apply");
  return failure();
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
