//===- SCFLoopUnrollPass.cpp - SCF Loop Unrolling Pass -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass walks over each scf.for operation in a function and unrolls
// innermost scf.for loops.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scf-loop-unroll"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_SCFLOOPUNROLL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Extract constant integer value from a Value.
static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

/// Returns `true` if forOp is innermost.
static bool isInnermostLoop(scf::ForOp forOp) {
  bool hasNestedLoop = false;
  forOp.getBody()->walk([&](scf::ForOp nestedForOp) {
    if (nestedForOp != forOp) {
      hasNestedLoop = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  forOp.getBody()->walk([&](scf::ForallOp nestedForallOp) {
    hasNestedLoop = true;
    return WalkResult::interrupt();
  });
  return !hasNestedLoop;
}

/// Helper function to decide whether a loop should be unrolled.
/// Checks loop bounds and unroll factor relations to make sure
/// unroll does not cause skipped or extra iterations.
static bool shouldUnrollLoop(scf::ForOp forOp, int64_t unrollFactor) {
  DBG("Analyzing loop: " << forOp.getLoc());

  // Extract constants. Skip if bounds or step not constant.
  auto lowerBoundOpt = getConstantIntValue(forOp.getLowerBound());
  auto upperBoundOpt = getConstantIntValue(forOp.getUpperBound());
  auto stepOpt = getConstantIntValue(forOp.getStep());
  if (!lowerBoundOpt || !upperBoundOpt || !stepOpt) {
    DBG("Failed to get constant bounds/step");
    return false;
  }

  int64_t lb = *lowerBoundOpt;
  int64_t ub = *upperBoundOpt;
  int64_t st = *stepOpt;

  if (st <= 0 || unrollFactor <= 0) {
    return false;
  }

  // Guard against negative (ub - lb).
  int64_t diff = ub - lb;
  if (diff <= 0) {
    return false;
  }

  // Equivalent to : (diff + st - 1) / st.
  int64_t tripCount = 1 + ((diff - 1) / st);

  DBG("Loop bounds: [" << lb << ", " << ub << ") step " << st);
  DBG("Trip count: " << tripCount);

  // If we don't generate a remainder/cleanup, require exact divisibility.
  if (tripCount < unrollFactor || tripCount % unrollFactor != 0)
    return false;

  if (unrollFactor > 16) {
    DBG("Unroll factor too large");
    return false;
  }

  // Check that there are no iter_args (loop-carried values).
  if (!forOp.getInitArgs().empty()) {
    DBG("Loop has iter_args (loop-carried values), skipping unroll");
    return false;
  }

  return true;
}

/// Helper function to perform the actual unrolling.
static LogicalResult unrollLoop(scf::ForOp forOp, OpBuilder &rewriter,
                                int64_t unrollFactor) {
  DBG("Unrolling loop at " << forOp.getLoc() << " by factor " << unrollFactor);
  Location loc = forOp.getLoc();
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  auto lowerBoundInt = *getConstantIntValue(lowerBound);
  auto upperBoundInt = *getConstantIntValue(upperBound);
  auto stepInt = *getConstantIntValue(step);

  // Calculate new step (step * unrollFactor)
  int64_t newStepInt = stepInt * unrollFactor;
  Value newStep = arith::ConstantIndexOp::create(rewriter, loc, newStepInt);

  // Create the new unrolled loop with larger step
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);

  auto newForOp = scf::ForOp::create(rewriter, loc, lowerBound, upperBound,
                                     newStep, forOp.getInitArgs());

  Block *newBody = newForOp.getBody();

  // Remove the automatically created yield terminator
  newBody->getTerminator()->erase();

  rewriter.setInsertionPointToStart(newBody);

  // Get the new induction variable
  Value newIV = newForOp.getInductionVar();

  // Clone the loop body `unrollFactor` times
  SmallVector<IRMapping> mappings(unrollFactor);
  for (int64_t i = 0; i < unrollFactor; ++i) {
    IRMapping &mapping = mappings[i];

    // Map the old induction variable to: newIV + i * step
    Value offset;
    if (i == 0) {
      offset = newIV;
    } else {
      Value iStep = arith::ConstantIndexOp::create(rewriter, loc, i * stepInt);
      offset = arith::AddIOp::create(rewriter, loc, newIV, iStep);
    }
    mapping.map(forOp.getInductionVar(), offset);

    // Map loop-carried values (iter_args)
    for (auto [oldArg, newArg] :
         llvm::zip(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs()))
      mapping.map(oldArg, newArg);

    // Clone all operations in the original loop body except the terminator
    Block *oldBody = forOp.getBody();
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }
  }

  // Handle the yield operation
  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> yieldValues;

  if (!oldYield.getResults().empty()) {
    // Use the mapping from the last unrolled iteration
    IRMapping &lastMapping = mappings[unrollFactor - 1];
    for (Value yieldVal : oldYield.getResults()) {
      yieldValues.push_back(lastMapping.lookupOrDefault(yieldVal));
    }
  }
  scf::YieldOp::create(rewriter, loc, yieldValues);

  // Replace the old loop with the new unrolled loop
  forOp->replaceAllUsesWith(newForOp.getResults());
  forOp->erase();
  DBG("Successfully unrolled loop");
  return success();
}

/// Pattern to match and unroll scf.for operations
struct SCFLoopUnrollPattern : public OpRewritePattern<scf::ForOp> {
  int64_t unrollFactor;

  SCFLoopUnrollPattern(MLIRContext *context, int64_t factor)
      : OpRewritePattern<scf::ForOp>(context), unrollFactor(factor) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    DBG("Candidate for loop unrolling:\n" << forOp);
    if (!shouldUnrollLoop(forOp, unrollFactor)) {
      return rewriter.notifyMatchFailure(forOp,
                                         "loop does not meet unroll criteria");
    }

    if (failed(unrollLoop(forOp, rewriter, unrollFactor))) {
      return rewriter.notifyMatchFailure(forOp, "failed to unroll loop");
    }
    DBG("Successfully unrolled loop");
    return success();
  }
};

struct SCFLoopUnrollPass : public ::impl::SCFLoopUnrollBase<SCFLoopUnrollPass> {

  explicit SCFLoopUnrollPass(const mlir::hexagon::SCFLoopUnrollOptions &options)
      : SCFLoopUnrollBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    DBG("Running SCF loop unroll pass with factor " << unrollFactor);
    // Collect only innermost loops.
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) {
      if (isInnermostLoop(forOp)) {
        forOps.push_back(forOp);
      }
    });
    DBG("Found " << forOps.size() << " innermost loops for unrolling");

    // Apply unrolling to each innermost loop.
    IRRewriter rewriter(context);
    for (scf::ForOp forOp : forOps) {
      // Skip if the operation was already erased
      if (!forOp->getBlock())
        continue;

      if (shouldUnrollLoop(forOp, unrollFactor)) {
        rewriter.setInsertionPoint(forOp);
        if (failed(unrollLoop(forOp, rewriter, unrollFactor))) {
          forOp.emitError("failed to unroll loop");
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createSCFLoopUnrollPass(const SCFLoopUnrollOptions &options) {
  return std::make_unique<SCFLoopUnrollPass>(options);
}
