//=====------- SlicingPass.cpp - Linalg Op Slicing Pass -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg op slicing, where an operation is broken down
// along its outermost loop into multiple parts. The results of these parts
// are concatenated at the end to create the original operation's expected
// result.
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

#define DEBUG_TYPE "hexagon-slicing"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONSLICING
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

#include <iostream>

namespace {

// Creates a slice of an operation
static SmallVector<Value>
createSlice(IRRewriter &rewriter, Location loc, TilingInterface op,
            ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
            const unsigned dimension, OpFoldResult size, OpFoldResult offset) {

  // Create modifable copies of sizes and offsets
  SmallVector<OpFoldResult> sizesCopy = llvm::to_vector(sizes);
  SmallVector<OpFoldResult> offsetsCopy = llvm::to_vector(offsets);

  // Update the size and offset for the specified dimension
  sizesCopy[dimension] = size;
  offsetsCopy[dimension] = offset;

  // Create the slice as a single tile
  DBG("-> Creating slice along dimension: " << dimension << " with offset ("
                                            << offset << ") , and size ("
                                            << size << ")\n");
  FailureOr<TilingResult> tilingResult =
      op.getTiledImplementation(rewriter, offsetsCopy, sizesCopy);

  // Check if tiling was successful
  if (failed(tilingResult)) {
    DBG("-> Failure in generating the slice\n");
    return {};
  }

  // Initialize a vector to store the results of the tiled operation
  SmallVector<Value> results(tilingResult->tiledValues.size());

  // Iterate over the tiled values and store each result in the corresponding
  // output index
  for (auto [index, result] : llvm::enumerate(tilingResult->tiledValues)) {
    DBG("-> Slice created for output["
        << index << "]:" << *(result.getDefiningOp()) << "\n");
    results[index] = result;
  }
  return results;
}

// Divides LoopRange (L) into K parts, ensuring that at least K-1 parts are
// powers of two. This approach is chosen instead of slicing the LoopRange
// into equal parts because equal slicing can result in multiple slices that
// are not powers of two. This can lead to issues during OneShotBufferize if
// each of these slices is padded and tiled separately in our pipeline. The
// current workaround for this issue is to add extra `bufferization.alloca`,
// which increases memory usage. Hence, this approach guarantees that only one
// part (the last part) may not be a power of two, which can trigger padding
// in the tiling pass.
SmallVector<int64_t> computeSliceSizes(int64_t L, uint64_t K) {
  if (L < K) {
    return {};
  }
  int64_t remainingRange = L;

  // Step1. Find the largest power of two that is smaller than the ideal slice
  // size
  auto idealSliceSize = L / K;
  int64_t largestPowerofTwo =
      (int64_t)1 << static_cast<int64_t>(std::floor(std::log2(idealSliceSize)));

  // Step 2. Create a vector to hold computed sizes,
  // Initialize all the vector elements to the
  // largest power of two computed in Step 1.
  SmallVector<int64_t> parts(K, largestPowerofTwo);

  // Step 3. Find the remaining value and try to redistribute it
  remainingRange -= (K * largestPowerofTwo);
  for (uint64_t i = 0; i < K && remainingRange > 0; ++i) {
    auto increaseCandidate = parts[i];
    if (remainingRange >= increaseCandidate) {
      parts[i] = 2 * increaseCandidate;
      remainingRange -= increaseCandidate;
    }
  }

  if (remainingRange > 0) {
    parts[K - 1] += remainingRange;
  }
  DBG("->Assigned slice sizes:\n");
  for (auto i = 0; i < parts.size(); ++i) {
    DBG("->Slice[" << i << "] = " << parts[i]);
  }

  return parts;
}

static LogicalResult sliceLinalgOp(linalg::LinalgOp op,
                                   uint64_t slicingFactor) {
  DBG("-> Slicing the op into " << slicingFactor << " parts\n");
  // Early exit.
  unsigned numLoops = op.getNumLoops();
  if (slicingFactor <= 1 || !op.hasPureTensorSemantics() ||
      !op->getNumResults() ||
      !isa<DestinationStyleOpInterface>(op.getOperation()) || numLoops < 1) {
    DBG("-> Slicing aborted. constraints not satisfied\n");
    return failure();
  }
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes[0] != utils::IteratorType::parallel) {
    DBG(" -> Outermost loop is not parallel. Slicing Aborted");
    return failure();
  }

  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  // Slicing along the outermost loop
  const unsigned dimension = 0;
  auto OuterLoopRange = op.getStaticLoopRanges()[0];
  auto opInterface = cast<TilingInterface>(op.getOperation());

  // Compute the Slice sizes based on the loop range and slicing factor
  auto sliceSizes = computeSliceSizes(OuterLoopRange, slicingFactor);
  if (sliceSizes.empty()) {
    DBG("->LoopRange is smaller than the slicingFactor. Slicing aborted.");
    return failure();
  }

  // Initialize a vector of vectors to store the final slices.
  // Each inner vector corresponds to the slices created for a single output.
  unsigned numOpOutputs = op.getNumDpsInits();
  SmallVector<SmallVector<Value>> finalSlices(numOpOutputs);

  // get the iteration space
  SmallVector<Range> iterationSpace = opInterface.getIterationDomain(rewriter);
  SmallVector<OpFoldResult> offsets = llvm::to_vector(llvm::map_range(
      iterationSpace, [](const Range &range) { return range.offset; }));
  SmallVector<OpFoldResult> sizes = llvm::to_vector(llvm::map_range(
      iterationSpace, [](const Range &range) { return range.size; }));

  int64_t sliceOffset = 0;
  for (int i = 0; i < slicingFactor; ++i) {
    int64_t sliceSize = sliceSizes[i];
    OpFoldResult sliceSizeAttr = rewriter.getIndexAttr(sliceSizes[i]);
    OpFoldResult sliceOffsetAttr = rewriter.getIndexAttr(sliceOffset);

    // Create the slice on the dimension for the computed offset and size
    auto opSlices = createSlice(rewriter, loc, opInterface, offsets, sizes,
                                dimension, sliceSizeAttr, sliceOffsetAttr);
    if (opSlices.empty()) {
      DBG("-> Failed to create slice. Slicing aborted.");
      return failure();
    }

    // Add the new slices to their correspoinding output vector in the
    // finalSlice
    for (int index = 0; index < numOpOutputs; index++) {
      finalSlices[index].push_back(opSlices[index]);
    }
    sliceOffset += sliceSize;
  }

  // For each output, create a `tensor.concat` operation to concatenate the
  // results of all the slices created for that output
  SmallVector<Value> concatOps(numOpOutputs);
  for (int index = 0; index < numOpOutputs; index++) {
    auto output = op.getDpsInitOperand(index)->get();
    auto concatType = cast<RankedTensorType>(output.getType());
    auto Slices = finalSlices[index];
    Value concatOp =
        rewriter.create<tensor::ConcatOp>(loc, concatType, 0, Slices);
    concatOps[index] = concatOp;
  }

  // replace the original op with the concat ops
  rewriter.replaceOp(op, concatOps);

  return success();
}

struct HexagonSlicingPass
    : public ::impl::HexagonSlicingBase<HexagonSlicingPass> {

  explicit HexagonSlicingPass(const HexagonSlicingOptions &options)
      : HexagonSlicingBase(options) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG(" Slicing candidate: " << op << "\n");
      if (succeeded(sliceLinalgOp(op, slicingFactor))) {
        DBG("-> Slicing succeeded\n");
      } else {
        DBG("-> Slicing failed\n");
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonSlicingPass(const HexagonSlicingOptions &options) {
  return std::make_unique<HexagonSlicingPass>(options);
}
