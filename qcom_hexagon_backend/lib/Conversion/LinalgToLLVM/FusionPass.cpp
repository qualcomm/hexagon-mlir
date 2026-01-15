//===- FusionPass.cpp - linalg tensor op fusion   -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass uses linalg-fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#define DEBUG_TYPE "hexagon-fusion"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONFUSION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

static constexpr unsigned MaxMultiUseFusionIterations = 3;

struct HexagonFusionPass : public ::impl::HexagonFusionBase<HexagonFusionPass> {
  explicit HexagonFusionPass(const HexagonFusionOptions &options)
      : HexagonFusionBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }
  void runOnOperation() override;
};

/// Return all op-operands in 'consumer' whose defining op is 'producer'.
static SmallVector<OpOperand *> getAllUsesInConsumer(Operation *producer,
                                                     Operation *consumer) {
  SmallVector<OpOperand *> allUses;
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      allUses.push_back(&opOperand);
    }
  }
  return allUses;
}

/// Find consumer that dominates all others consumers.
static std::optional<OpOperand *> getDominantConsumer(Operation *producer,
                                                      DominanceInfo &domInfo) {
  auto consumers = producer->getUses();
  for (OpOperand &consumer : consumers) {
    Operation *consumerOp = consumer.getOwner();
    bool dominatesAllUsers = true;
    for (OpOperand &other : consumers) {
      Operation *otherConsumerOp = other.getOwner();
      if (!domInfo.dominates(consumerOp, otherConsumerOp)) {
        dominatesAllUsers = false;
        break;
      }
    }
    if (dominatesAllUsers) {
      return &consumer;
    }
  }
  return std::nullopt;
}

// Get the first consumer op-operand that uses this producer.
static OpOperand *getFirstUseInConsumer(Operation *producer,
                                        Operation *consumer) {
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      return &opOperand;
    }
  }
  return nullptr;
}

/// Fuse two linalg generic using builtin elemwise-op-fusion.
static LogicalResult doMultiUseFusion(Operation *consumer,
                                      llvm::SetVector<Operation *> &producers,
                                      RewriterBase &rewriter) {
  assert(consumer && "null consumer");

  SmallVector<Operation *> producersVec = llvm::to_vector(producers);
  // def-use
  mlir::computeTopologicalSorting(producersVec);

  Operation *consumerOp = consumer;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *producerOp : llvm::reverse(producersVec)) {
    while (OpOperand *fusedOperand =
               getFirstUseInConsumer(producerOp, consumerOp)) {
      rewriter.setInsertionPoint(consumerOp);
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, fusedOperand);

      if (failed(fusionResult)) {
        return rewriter.notifyMatchFailure(consumerOp,
                                           "failed to fuse with producer");
      }

      for (auto replacement : fusionResult->replacements) {
        // Find uses of 'first' and replace with 'second'
        rewriter.replaceUsesWithIf(
            replacement.first, replacement.second, [&](OpOperand &use) {
              return use.getOwner() != fusionResult->fusedOp &&
                     producers.count(use.getOwner()) == 0;
            });
      }
      consumerOp = fusionResult->fusedOp;
      if (failed(cast<linalg::GenericOp>(consumerOp).verify())) {
        return consumerOp->emitOpError("failed to verify op");
      }
    }
  }
  return success();
}

/// Drive a single iteration of multi-use fusion.
static FailureOr<unsigned> multiUseFusion(MLIRContext *context,
                                          Operation *funcOp,
                                          DominanceInfo &domInfo) {
  OpBuilder builder(context);
  // map from consumer to fusable producers.
  llvm::MapVector<Operation *, llvm::SetVector<Operation *>> fusedOps;

  // producer to consumer map.
  DenseMap<Operation *, Operation *> prodToConsMap;

  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](linalg::GenericOp consumer) {
        if (consumer.getNumLoops() != consumer.getNumParallelLoops()) {
          return;
        }
        Operation *fusableProducer = nullptr;
        for (OpOperand &operand : consumer->getOpOperands()) {
          auto producer = dyn_cast_or_null<linalg::GenericOp>(
              operand.get().getDefiningOp());

          if (!producer || prodToConsMap.count(producer) ||
              (producer->getBlock() != consumer->getBlock()) ||
              !linalg::areElementwiseOpsFusable(&operand)) {
            continue;
          }
          if (producer.getNumLoops() != producer.getNumParallelLoops() ||
              consumer.getNumLoops() != producer.getNumLoops()) {
            continue;
          }
          // check that this consumer is dominant over all others.
          std::optional<OpOperand *> fusableUse =
              getDominantConsumer(producer, domInfo);
          if (!fusableUse || fusableUse.value()->getOwner() != consumer) {
            continue;
          }
          // check all uses from this producer is elem-wise.
          if (llvm::any_of(getAllUsesInConsumer(producer, consumer),
                           [&](OpOperand *use) {
                             return !linalg::areElementwiseOpsFusable(use);
                           })) {
            continue;
          }
          fusableProducer = producer;
          break;
        }
        if (!fusableProducer)
          return;

        llvm::SetVector<Operation *> &fusedOpSet = fusedOps[consumer];
        fusedOpSet.insert(fusableProducer);
        prodToConsMap[fusableProducer] = consumer;
        return;
      });
  if (fusedOps.empty()) {
    return 0;
  }
  // Fuse producer consumer
  IRRewriter rewriter(context);
  for (auto it = fusedOps.rbegin(), ie = fusedOps.rend(); it != ie; ++it) {
    if (failed(doMultiUseFusion(it->first, it->second, rewriter))) {
      return funcOp->emitOpError("failed multi use fusion");
    }
  }
  // clean up.
  RewritePatternSet fusionPatterns(context);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
    return funcOp->emitOpError("multi use producer -> consumer fusion failed");
  }
  return fusedOps.size();
}

void HexagonFusionPass::runOnOperation() {
  auto funcOp = getOperation();
  auto context = &getContext();
  RewritePatternSet fusionPatterns(context);

  auto controlElemwiseFn = [&](OpOperand *operand) {
    Operation *producer = operand->get().getDefiningOp();
    return linalg::areElementwiseOpsFusable(operand) &&
           (producer->hasOneUse() || fusionAllowRecompute);
  };
  linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                               controlElemwiseFn);

  linalg::ControlFusionFn fuseByExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        return producer->hasOneUse();
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(fusionPatterns,
                                                    fuseByExpansionControlFn);

  // We don't want multiple copies of constant created.
  auto constantFoldControlFn = [](OpOperand *fusedOperand) {
    Operation *producer = fusedOperand->get().getDefiningOp();
    return producer->hasOneUse();
  };
  linalg::populateConstantFoldLinalgOperations(fusionPatterns,
                                               constantFoldControlFn);

  // Canonicalization patterns
  affine::AffineApplyOp::getCanonicalizationPatterns(fusionPatterns, context);
  linalg::GenericOp::getCanonicalizationPatterns(fusionPatterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(fusionPatterns, context);

  tensor::populateFoldTensorEmptyPatterns(fusionPatterns);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(fusionPatterns, context);
  context->getLoadedDialect<linalg::LinalgDialect>()
      ->getCanonicalizationPatterns(fusionPatterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(fusionPatterns);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.setMaxIterations(GreedyRewriteConfig::kNoLimit);
  if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns),
                                   rewriteConfig))) {
    funcOp->emitError("failed to apply fusion patterns");
    return signalPassFailure();
  }

  if (fusionDoMultiUse) {
    unsigned nIter = 0;
    while (nIter++ < MaxMultiUseFusionIterations) {
      auto &domInfo = getAnalysis<DominanceInfo>();
      auto numCandidates = multiUseFusion(context, funcOp, domInfo);
      if (failed(numCandidates)) {
        funcOp->emitError("multi use fusion failed");
        return signalPassFailure();
      }
      if (!numCandidates.value())
        break;
    }
  }
  return;
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonFusionPass(const HexagonFusionOptions &options) {
  return std::make_unique<HexagonFusionPass>(options);
}
