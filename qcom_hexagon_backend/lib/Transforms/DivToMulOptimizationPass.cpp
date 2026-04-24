//===- DivToMulOptimizationPass.cpp - Optimize Div to Mul -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes division operations in linalg.generic by converting them
// to multiplications when the divisor is broadcasted. It handles two broadcast
// patterns: (1) linalg.fill operations that broadcast scalars to tensors, and
// (2) linalg.generic operations that broadcast lower-rank tensors to higher
// dimensions. The optimization computes the reciprocal (1/x) at the original
// precision before broadcasting, then replaces division with multiplication.
// This transformation is beneficial on hardware accelerators where
// multiplication is significantly faster than division.
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "div-to-mul-optimization"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;

#define GEN_PASS_DEF_DIVTOMULOPTIMIZATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Helper to replace div with mul after computing reciprocal.
static void replaceDivWithMul(linalg::GenericOp op, arith::DivFOp divOp,
                              Value broadcastedInverse, unsigned argNum,
                              PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.modifyOpInPlace(
      op, [&]() { op.setOperand(argNum, broadcastedInverse); });
  rewriter.setInsertionPoint(divOp);
  Value mulOp = arith::MulFOp::create(rewriter, divOp.getLoc(), divOp.getLhs(),
                                      divOp.getRhs());
  rewriter.replaceOp(divOp, mulOp);
}

/// Helper to handle div opt. when divisor comes from a fillOp.
/// Transforms: x[][] / filled(y) -> x[][] * filled(1/y)
static LogicalResult handleFillDivisor(linalg::GenericOp op,
                                       arith::DivFOp divOp, Value divisorInput,
                                       unsigned argNum,
                                       PatternRewriter &rewriter) {
  auto fillOp = divisorInput.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return failure();

  DBG("Found fill operation as divisor");
  Value fillValue = fillOp.getInputs()[0];
  auto floatType = dyn_cast<FloatType>(fillValue.getType());
  if (!floatType)
    return rewriter.notifyMatchFailure(op, "fill value not float type");

  OpBuilder::InsertionGuard guard(rewriter);

  // Create reciprocal: 1/x
  rewriter.setInsertionPoint(fillOp);
  Value one = arith::ConstantOp::create(rewriter, fillOp.getLoc(),
                                        rewriter.getFloatAttr(floatType, 1.0));
  Value reciprocal =
      arith::DivFOp::create(rewriter, fillOp.getLoc(), one, fillValue);

  // Create new fillOp with reciprocal value
  Value broadcastedInverse =
      linalg::FillOp::create(rewriter, fillOp.getLoc(), reciprocal,
                             fillOp.getOutputs()[0])
          .getResult(0);

  // Use common helper to rewrite original generic.division.
  replaceDivWithMul(op, divOp, broadcastedInverse, argNum, rewriter);
  DBG("Successfully optimized fill-div to mul");
  return success();
}

/// Helper to handle div opt. when divisor comes from a linalg.generic
/// that is broadcasting a tensor to a higher dimension.
static LogicalResult handleLinGenDivisor(linalg::GenericOp op,
                                         arith::DivFOp divOp,
                                         Value divisorInput, unsigned argNum,
                                         PatternRewriter &rewriter) {
  // Check if divisorInput comes from a linalg.generic operation
  auto broadcastOp = divisorInput.getDefiningOp<linalg::GenericOp>();
  if (!broadcastOp)
    return failure();
  DBG("Found linalg.generic operation as divisor");

  // Check: broadcastOp is one input and one output.
  if (broadcastOp.getNumDpsInputs() != 1 || broadcastOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "broadcast needs to be single in out.");

  // Check: body is just a yield.
  Block *broadcastBody = broadcastOp.getBody();
  if (std::distance(broadcastBody->begin(), broadcastBody->end()) != 1)
    return rewriter.notifyMatchFailure(op, "broadcast body must be only yield");

  auto broadcastYield =
      dyn_cast<linalg::YieldOp>(broadcastBody->getTerminator());
  if (!broadcastYield || broadcastYield.getNumOperands() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "broadcast body must be single yield");

  // The yield value should be a block argument.
  auto yieldedArg = dyn_cast<BlockArgument>(broadcastYield.getOperand(0));
  if (!yieldedArg)
    return rewriter.notifyMatchFailure(op, "broadcast must yield its input");
  if (yieldedArg.getArgNumber() != 0)
    return rewriter.notifyMatchFailure(op, "input is strangely not yielded");

  // Check: broadcastOp input is lower dimension than output
  Value broadcastInput = broadcastOp.getDpsInputOperand(0)->get();
  Value broadcastOutput = broadcastOp.getDpsInitOperand(0)->get();

  auto inputType = dyn_cast<RankedTensorType>(broadcastInput.getType());
  auto outputType = dyn_cast<RankedTensorType>(broadcastOutput.getType());

  if (!inputType || !outputType)
    return rewriter.notifyMatchFailure(
        op, "broadcast operands must be ranked tensors");

  if (inputType.getRank() >= outputType.getRank())
    return rewriter.notifyMatchFailure(
        op, "broadcast input must be lower rank than output");

  // Check: the maps of input and output are projected permutation
  SmallVector<AffineMap> indexingMaps = broadcastOp.getIndexingMapsArray();
  if (indexingMaps.size() != 2)
    return rewriter.notifyMatchFailure(op,
                                       "broadcast must have 2 indexing maps");

  if (!indexingMaps[0].isProjectedPermutation() ||
      !indexingMaps[1].isProjectedPermutation())
    return rewriter.notifyMatchFailure(op,
                                       "maps must be projected permutations");
  DBG("Verified linalg generic is a pure broadcast op");

  // Get the element type for creating the reciprocal.
  auto elementType = dyn_cast<FloatType>(inputType.getElementType());
  if (!elementType)
    return rewriter.notifyMatchFailure(
        op, "broadcast input must have float element type");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(broadcastOp);

  // Create a generic op to compute reciprocal.
  Value reciprocalOutput = tensor::EmptyOp::create(
      rewriter, broadcastOp.getLoc(), inputType.getShape(), elementType);

  // Create indexing maps for the reciprocal operation (identity maps)
  SmallVector<AffineMap> reciprocalMaps;
  unsigned inputRank = inputType.getRank();
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(inputRank, rewriter.getContext());
  reciprocalMaps.push_back(identityMap); // input map
  reciprocalMaps.push_back(identityMap); // output map

  SmallVector<utils::IteratorType> iteratorTypes(inputRank,
                                                 utils::IteratorType::parallel);

  // Create the reciprocal generic op
  auto reciprocalOp = linalg::GenericOp::create(
      rewriter, broadcastOp.getLoc(), reciprocalOutput.getType(),
      ValueRange{broadcastInput}, ValueRange{reciprocalOutput}, reciprocalMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(elementType, 1.0));
        Value reciprocal = arith::DivFOp::create(b, loc, one, args[0]);
        linalg::YieldOp::create(b, loc, reciprocal);
      });

  // Clone the broadcast op with reciprocal as input
  Value reciprocalResult = reciprocalOp.getResult(0);

  auto clonedBroadcastOp = linalg::GenericOp::create(
      rewriter, broadcastOp.getLoc(), broadcastOp.getResultTypes(),
      ValueRange{reciprocalResult}, broadcastOp.getDpsInits(),
      broadcastOp.getIndexingMapsArray(), broadcastOp.getIteratorTypesArray(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        linalg::YieldOp::create(b, loc, args[0]);
      });

  Value broadcastedInverse = clonedBroadcastOp.getResult(0);

  // Use common helper to rewrite original generic division
  replaceDivWithMul(op, divOp, broadcastedInverse, argNum, rewriter);
  DBG("Successfully optimized linalg.generic broadcast-div to mul");
  return success();
}

/// Replace broadcasted division with multiplication.
struct BroadcastDivToMulOptimizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    DBG("Candidate for div-to-mul opt:\n" << op);

    Block *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "must have single yield");

    // Try to find the division operation in the body.
    arith::DivFOp divOp = nullptr;
    Value yieldValue = yieldOp.getOperand(0);

    // Case 1: Direct division followed by yield
    // Pattern: divf -> yield
    if (std::distance(body->begin(), body->end()) == 2) {
      divOp = dyn_cast<arith::DivFOp>(&body->front());
      if (divOp && yieldValue == divOp.getResult()) {
        DBG("Found direct div-yield pattern");
      } else {
        divOp = nullptr;
      }
    }

    // Case 2: Division with type conversions
    // Pattern: extf -> extf -> divf -> truncf -> yield
    if (!divOp && std::distance(body->begin(), body->end()) == 5) {
      auto truncOp = yieldValue.getDefiningOp<arith::TruncFOp>();
      if (truncOp) {
        divOp = truncOp.getOperand().getDefiningOp<arith::DivFOp>();
        if (divOp) {
          // Verify the dividend and divisor come from extf operations
          auto lhsExtf = divOp.getLhs().getDefiningOp<arith::ExtFOp>();
          auto rhsExtf = divOp.getRhs().getDefiningOp<arith::ExtFOp>();
          if (lhsExtf && rhsExtf && lhsExtf != rhsExtf &&
              lhsExtf.getOperand() != rhsExtf.getOperand()) {
            auto lhsBlockArg = dyn_cast<BlockArgument>(lhsExtf.getOperand());
            auto rhsBlockArg = dyn_cast<BlockArgument>(rhsExtf.getOperand());
            if (lhsBlockArg && rhsBlockArg &&
                lhsBlockArg.getArgNumber() != rhsBlockArg.getArgNumber()) {
              // all check pass.
              DBG("Found extf-extf-divf-truncf-yield pattern");
            } else {
              divOp = nullptr;
            }
          } else {
            divOp = nullptr;
          }
        }
      }
    }

    if (!divOp)
      return rewriter.notifyMatchFailure(op,
                                         "no supported division pattern found");

    // Dividend and divisor must not be the same
    if (divOp.getLhs() == divOp.getRhs())
      return rewriter.notifyMatchFailure(op,
                                         "dividend and divisor are the same");

    // Find the divisor block argument (may be through extf)
    Value divisorValue = divOp.getRhs();
    auto divisorExtf = divisorValue.getDefiningOp<arith::ExtFOp>();
    if (divisorExtf)
      divisorValue = divisorExtf.getOperand();

    auto divisorArg = dyn_cast<BlockArgument>(divisorValue);
    if (!divisorArg)
      return rewriter.notifyMatchFailure(op, "divisor not a block-arg");

    unsigned argNum = divisorArg.getArgNumber();
    if (argNum >= op.getNumDpsInputs())
      return rewriter.notifyMatchFailure(op, "divisor not an input");
    DBG("Passes division checks!");

    Value divisorInput = op.getDpsInputOperand(argNum)->get();
    DBG("Divisor defining op is:\n" << divisorInput);

    // Handle fillOp divisor
    if (succeeded(handleFillDivisor(op, divOp, divisorInput, argNum, rewriter)))
      return success();

    // Handle linalg.generic broadcaster of divisor
    if (succeeded(
            handleLinGenDivisor(op, divOp, divisorInput, argNum, rewriter)))
      return success();

    return rewriter.notifyMatchFailure(op, "divisor type not supported yet");
  }
};

struct DivToMulOptimizationPass
    : public ::impl::DivToMulOptimizationBase<DivToMulOptimizationPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<BroadcastDivToMulOptimizationPattern>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createDivToMulOptimizationPass() {
  return std::make_unique<DivToMulOptimizationPass>();
}
