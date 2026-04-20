//===- EraseVectorToTensorWritebackPass.cpp - Erase Unnecessary Writeback -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes IR where vectors are written back to tensors and yielded
// in scf.for loops, only to be immediately read back as vectors after the loop.
// When the output tensor has no other uses, the loop can directly yield vector.
//
// Example transformation:
//   Before:
//     %init = vector.transfer_write %vec, %tensor : vector<32xf32>,
//     tensor<32xf32> %result = scf.for %i = ... iter_args(%arg = %init) ->
//     (tensor<32xf32>) {
//       ...
//       %new_vec = ... : vector<32xf32>
//       %write = vector.transfer_write %new_vec, %empty : vector<32xf32>,
//       tensor<32xf32> scf.yield %write : tensor<32xf32>
//     }
//     %final_vec = vector.transfer_read %result : tensor<32xf32>,
//     vector<32xf32>
//
//   After:
//     %result = scf.for %i = ... iter_args(%arg = %vec) -> (vector<32xf32>) {
//       ...
//       %new_vec = ... : vector<32xf32>
//       scf.yield %new_vec : vector<32xf32>
//     }
//     // %result is already a vector, no transfer_read needed
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "erase-vector-to-tensor-writeback"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::vector;

#define GEN_PASS_DEF_ERASEVECTORTOTENSORWRITEBACK
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Returns 'true' if forOp yield is a valid candidate.
static bool shouldTransformForOp(scf::ForOp forOp) {

  // 1. Check result out of the forOp for validity.
  // The result must be a ranked tensor type with exactly one use.
  if (forOp.getNumResults() != 1)
    return false;

  Value result = forOp.getResult(0);
  auto tensorType = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorType || !result.hasOneUse())
    return false;

  // The single user must be a vector.transfer_read of complete
  // contents from tensor to vector.
  Operation *user = *result.getUsers().begin();
  auto transferReadOp = dyn_cast<TransferReadOp>(user);
  if (!transferReadOp)
    return false;

  auto vectorType = dyn_cast<VectorType>(transferReadOp.getType());
  if (!vectorType)
    return false;

  if (tensorType.getShape() != vectorType.getShape()) {
    return false;
  }

  // 2. Check the tensor passed in as iter arg for validity.
  // The tensor passed in must have exactly one use which is
  // to move of complete contents to vector.
  if (forOp.getRegionIterArgs().empty())
    return false;
  BlockArgument iterArg = forOp.getRegionIterArgs()[0];
  auto iterArgTensorType = dyn_cast<RankedTensorType>(iterArg.getType());
  if (!iterArgTensorType)
    return false;
  if (!iterArg.hasOneUse()) {
    return false;
  }
  Operation *iterArgUser = iterArg.use_begin()->getOwner();
  auto iterArgTransferRead = dyn_cast<TransferReadOp>(iterArgUser);
  if (!iterArgTransferRead)
    return false;

  auto iterArgVectorType = dyn_cast<VectorType>(iterArgTransferRead.getType());
  if (!iterArgVectorType)
    return false;

  if (iterArgTensorType.getShape() != iterArgVectorType.getShape())
    return false;

  // 3. Check the yielded value comes from complete vector.transfer_write.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldedValue = yieldOp.getOperand(0);

  auto transferWriteOp = yieldedValue.getDefiningOp<TransferWriteOp>();
  if (!transferWriteOp)
    return false;

  Value inputVector = transferWriteOp.getVector();
  auto inputVectorType = dyn_cast<VectorType>(inputVector.getType());
  if (!inputVectorType)
    return false;

  auto resultTensorType =
      dyn_cast<RankedTensorType>(transferWriteOp.getType(0));
  if (!resultTensorType)
    return false;

  if (inputVectorType.getShape() != resultTensorType.getShape())
    return false;

  return true;
}

/// Transform a scf.for to yield vectors instead of tensors.
static LogicalResult transformForOp(scf::ForOp forOp) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Get the single init arg and its tensor type.
  Value initArg = forOp.getInitArgs()[0];
  auto tensorType = cast<RankedTensorType>(initArg.getType());

  // Extract the vector from the initial tensor.
  builder.setInsertionPoint(forOp);
  SmallVector<OpFoldResult> indices;
  for (unsigned i = 0; i < tensorType.getRank(); ++i)
    indices.push_back(builder.getIndexAttr(0));

  auto vectorType =
      VectorType::get(tensorType.getShape(), tensorType.getElementType());
  SmallVector<Value> indicesValues =
      getValueOrCreateConstantIndexOp(builder, loc, indices);

  // Create padding value (zero for the element type).
  Value padding = arith::ConstantOp::create(
      builder, loc, tensorType.getElementType(),
      builder.getZeroAttr(tensorType.getElementType()));

  // create the vector.transer_read above the scf.for
  auto transferReadOp = TransferReadOp::create(
      builder, loc, vectorType, initArg, indicesValues, padding, std::nullopt);
  Value vectorValue = transferReadOp.getResult();

  // Create new scf.for with vector iter_arg
  auto newForOp = scf::ForOp::create(
      builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), ValueRange{vectorValue},
      [](OpBuilder &, Location, Value, ValueRange) {});

  // Clone the loop body
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  BlockArgument oldIterArg = forOp.getRegionIterArgs()[0];
  BlockArgument newIterArg = newForOp.getRegionIterArgs()[0];
  mapping.map(oldIterArg, newIterArg);

  builder.setInsertionPointToStart(newForOp.getBody());
  for (auto &op : forOp.getBody()->without_terminator()) {
    // Skip cloning vector.transfer_read that reads from the iter_arg
    if (auto transferReadOp = dyn_cast<TransferReadOp>(&op)) {
      // Check if the first operand (base/source) is the iter_arg
      if (transferReadOp.getOperand(0) == oldIterArg) {
        // Replace with the vector iter_arg directly
        mapping.map(transferReadOp.getResult(), newIterArg);
        continue;
      }
    }
    builder.clone(op, mapping);
  }

  // Update the yield operation
  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldedValue = oldYield.getOperand(0);
  Value mappedYieldedValue = mapping.lookupOrDefault(yieldedValue);

  auto transferWriteOp = mappedYieldedValue.getDefiningOp<TransferWriteOp>();
  if (!transferWriteOp) {
    DBG("Failed to find transfer_write for yielded value");
    return failure();
  }

  Value vectorToYield = transferWriteOp.getVector();
  scf::YieldOp::create(builder, loc, vectorToYield);

  // Erase the transfer_write operation if it's no longer needed
  if (transferWriteOp.use_empty())
    transferWriteOp.erase();

  // Replace uses of the old for loop result
  Value oldResult = forOp.getResult(0);
  Value newResult = newForOp.getResult(0);

  // Replace transfer_read users with the vector result directly
  SmallVector<TransferReadOp> transferReadsToReplace;
  for (auto user : oldResult.getUsers()) {
    if (auto transferReadOp = dyn_cast<TransferReadOp>(user)) {
      transferReadsToReplace.push_back(transferReadOp);
    }
  }

  for (auto transferReadOp : transferReadsToReplace) {
    transferReadOp.replaceAllUsesWith(newResult);
    transferReadOp.erase();
  }

  forOp.erase();
  DBG("Successfully transformed scf.for");
  return success();
}

struct EraseVectorToTensorWritebackPass
    : public ::impl::EraseVectorToTensorWritebackBase<
          EraseVectorToTensorWritebackPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    DBG("Running EraseVectorToTensorWriteback pass on function: "
        << funcOp.getName());

    SmallVector<scf::ForOp> forOpsToTransform;

    // Collect all scf.for ops that match our pattern
    funcOp.walk([&](scf::ForOp forOp) {
      if (shouldTransformForOp(forOp)) {
        forOpsToTransform.push_back(forOp);
      }
    });

    // Transform the collected for ops
    for (auto forOp : forOpsToTransform) {
      if (failed(transformForOp(forOp))) {
        DBG("Failed to transform scf.for op");
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createEraseVectorToTensorWritebackPass() {
  return std::make_unique<EraseVectorToTensorWritebackPass>();
}
