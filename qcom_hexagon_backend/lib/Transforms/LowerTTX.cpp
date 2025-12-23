//===- LowerTTXPass.cpp: convert ttx dialect ops to mlir core.    ------======//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of ttx dialect ops e.g. ttx.cumsum to some
// combination of linalg, scf and other mlir core ops.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "lower-ttx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERTTX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LowerTTXPass : public ::impl::LowerTTXBase<LowerTTXPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ttx::TTXDialect>();
  }

  void runOnOperation() override;
};

// Get dim of tensor as const (static) or val.
static Value getAsDimOpOrConst(OpBuilder &builder, Location loc, Value tensor,
                               unsigned dim) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  Value val;
  assert(tensorType && "expected ranked tensor type");
  if (tensorType.isDynamicDim(dim)) {
    val = builder.create<tensor::DimOp>(loc, tensor, dim);
  } else {
    val =
        builder.create<arith::ConstantIndexOp>(loc, tensorType.getShape()[dim]);
  }
  return val;
}

//  Lower `ttx.scan` to sequence of mlir-core ops.
struct LowerScan : public OpRewritePattern<ttx::ScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttx::ScanOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    int64_t axis = op.getAxis();
    bool reverse = op.getReverse();
    int64_t rank = op.getRank();

    // corner case - just forward input to output.
    if (rank == 0) {
      rewriter.replaceAllUsesWith(op.getResult(), input);
      rewriter.eraseOp(op);
      return success();
    }

    // Create loop-body that forwards input args.
    auto buildScalarBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                               ValueRange args) -> scf::ValueVector {
      assert(ivs.size() == 1 && args.size() == 2 &&
             "expected one-deep loop with two iter-args");
      Value acc = args[0];
      Value input = args[1];
      return {acc, input};
    };

    // 1-D case. This is common and the likely only case.
    if (rank == 1) {
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value size = getAsDimOpOrConst(rewriter, loc, input, 0);

      // extract accumulator initial value.
      Value accInitVal;
      if (!reverse) {
        accInitVal =
            rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{zeroIdx});
      } else {
        auto sub = rewriter.create<arith::SubIOp>(loc, size, oneIdx);
        accInitVal =
            rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{sub});
      }

      // Create scf-loop.
      SmallVector<Value> iterArgs{accInitVal, input};
      scf::LoopNest loopNest = scf::buildLoopNest(
          rewriter, loc, {oneIdx}, {size}, {oneIdx}, iterArgs, buildScalarBody);
      auto loop = loopNest.loops[0];
      rewriter.replaceAllUsesWith(op.getResult(), loop.getResults()[1]);

      Block *opBody = &(op.getBody().front());
      Block *forBody = loop.getBody();
      Operation *forYield = forBody->getTerminator();

      auto forArgs = forBody->getArguments();
      Value iv = forArgs[0];
      Value acc = forArgs[1];
      Value tensor = forArgs[2];

      // Move scan op's body to loop body and erase op.
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(forYield);
      Value curr;
      if (!reverse) {
        curr = rewriter.create<tensor::ExtractOp>(loc, tensor, ValueRange{iv});
      } else {
        auto sizeMinusOne = rewriter.create<arith::SubIOp>(loc, size, oneIdx);
        auto revIdx = rewriter.create<arith::SubIOp>(loc, sizeMinusOne, iv);
        curr =
            rewriter.create<tensor::ExtractOp>(loc, tensor, ValueRange{revIdx});
      }
      rewriter.inlineBlockBefore(opBody, forBody, Block::iterator(forYield),
                                 {acc, curr});
      rewriter.eraseOp(op);

      // Forward scan.return yield value to output tensor and scf.yield.
      Block::iterator it = std::prev(forBody->end(), 2);
      auto scanReturn = dyn_cast<ttx::ScanReturnOp>(*it);
      assert(scanReturn && "second last op is not a scan.return");
      Value yieldVal = scanReturn.getYieldValue();
      scanReturn->erase();
      rewriter.setInsertionPoint(forBody->getTerminator());
      Value nextTensor;
      if (!reverse) {
        nextTensor = rewriter.create<tensor::InsertOp>(loc, yieldVal, tensor,
                                                       ValueRange{iv});
      } else {
        auto sizeMinusOne = rewriter.create<arith::SubIOp>(loc, size, oneIdx);
        auto revIdx = rewriter.create<arith::SubIOp>(loc, sizeMinusOne, iv);
        nextTensor = rewriter.create<tensor::InsertOp>(loc, yieldVal, tensor,
                                                       ValueRange{revIdx});
      }
      rewriter.setInsertionPoint(forBody->getTerminator());
      rewriter.create<scf::YieldOp>(loc, ValueRange{yieldVal, nextTensor});
      forBody->getTerminator()->erase();
      return success();
    }
    assert(false && "n-D scan lowering is not currently implemented");
    return failure();
  }
};

struct LowerCumSum : public OpRewritePattern<ttx::CummulativeSumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttx::CummulativeSumOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto axis = op.getAxis();
    auto input = op.getInput();
    auto rank = op.getRank();

    // corner case
    if (rank == 0) {
      rewriter.replaceAllUsesWith(op.getResult(0), input);
      rewriter.eraseOp(op);
      return success();
    }

    // add prefix to current value.
    auto buildAccumulator = [](OpBuilder &builder, Location loc, Type elType,
                               Value a, Value b) -> Value {
      if (elType.isFloat()) {
        return builder.create<arith::AddFOp>(loc, a, b).getResult();
      }
      return builder.create<arith::AddIOp>(loc, a, b).getResult();
    };

    // build loop body performing the scalar accumulation.
    auto buildScalarBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                               ValueRange args) -> scf::ValueVector {
      assert(ivs.size() == 1 && args.size() == 2 &&
             "expected one-deep loop with two iter-args");
      Value acc = args[0];
      Value input = args[1];
      Value idx = ivs[0];
      auto thisVal =
          builder.create<tensor::ExtractOp>(loc, input, ValueRange{idx});
      Value accNext =
          buildAccumulator(builder, loc, acc.getType(), acc, thisVal);

      auto inputNext = builder.create<tensor::InsertOp>(loc, accNext, input,
                                                        ValueRange{idx});
      return {accNext, inputNext};
    };

    // 1-D cumsum case. This is the common case and so we handle it
    // separately to enable writing it as optimized version.
    if (rank == 1) {
      Value lbs = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value ubs = getAsDimOpOrConst(rewriter, loc, input, 0);
      Value steps = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto accInitVal =
          rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{zeroIdx});

      SmallVector<Value> iterArgs{accInitVal, input};
      scf::LoopNest loopNest = scf::buildLoopNest(
          rewriter, loc, {lbs}, {ubs}, {steps}, iterArgs, buildScalarBody);

      auto loop = loopNest.loops[0];
      rewriter.replaceAllUsesWith(op.getResult(0), loop.getResults()[1]);
      rewriter.eraseOp(op);
      return success();
    }

    // General n-D cumsum case: input is n-D tensor and accumulation
    // is along the `axis` dimension.
    assert(rank > 1 && "'rank <= 1' expected at this stage");
    IntegerAttr zeroAttr = rewriter.getI64IntegerAttr(0);
    IntegerAttr oneAttr = rewriter.getI64IntegerAttr(1);

    SmallVector<OpFoldResult> zeroOffsets(rank, zeroAttr);
    SmallVector<OpFoldResult> oneStrides(rank, oneAttr);

    auto sizes = tensor::getMixedSizes(rewriter, loc, input);
    auto accSizes = sizes;
    accSizes[axis] = oneAttr;

    auto init = op.getOutput();
    auto inputType = dyn_cast<mlir::RankedTensorType>(input.getType());
    auto initType = dyn_cast<mlir::RankedTensorType>(init.getType());
    assert(inputType && initType && inputType == initType &&
           "expect input and output to be identical ranked tensors");

    auto accInitVal = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, zeroOffsets, accSizes, oneStrides);
    Value lbs = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ubs = getAsDimOpOrConst(rewriter, loc, input, axis);
    Value steps = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto buildTensorBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                               ValueRange args) -> scf::ValueVector {
      assert(ivs.size() == 1 && args.size() == 2 &&
             "expected 1-deep loop with two ivs");
      Value acc = args[0];
      Value input = args[1];
      auto thisOffsets = zeroOffsets;
      thisOffsets[axis] = ivs[0];

      auto thisVal = rewriter
                         .create<tensor::ExtractSliceOp>(
                             loc, input, thisOffsets, accSizes, oneStrides)
                         .getResult();

      // extract is at least 1-D tensor.
      auto elType = cast<RankedTensorType>(thisVal.getType()).getElementType();
      Value newAcc = buildAccumulator(builder, loc, elType, acc, thisVal);
      auto newInput = rewriter.create<tensor::InsertSliceOp>(
          loc, newAcc, input, thisOffsets, accSizes, oneStrides);
      return {newAcc, newInput};
    };

    SmallVector<Value> iterArgs{accInitVal, input};
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, {lbs}, {ubs}, {steps}, iterArgs, buildTensorBody);
    auto loop = loopNest.loops[0];
    rewriter.replaceAllUsesWith(op.getResult(0), loop.getResults()[1]);
    rewriter.eraseOp(op);
    return success();
  }
};

void LowerTTXPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerCumSum>(patterns.getContext());
  patterns.add<LowerScan>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createLowerTTXPass() {
  return std::make_unique<LowerTTXPass>();
}
