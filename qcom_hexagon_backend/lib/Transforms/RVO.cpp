//===- RVO.cpp - output copy elision (return value optimization) pass ----====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements form of `return value optimization`. Instead of storing
// the results into a local buffer and copying contents to output buffer before
// return, this pass transforms the IR so that `eliminate empty tensor` and
// `one-shot bufferizer` find it possible to do output copy elision.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-return-value-optimization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONRVO
#include "hexagon/Transforms/Passes.h.inc"

namespace {
using MaterializeOpTy = bufferization::MaterializeInDestinationOp;
using ReinterpretOpTy = memref::ReinterpretCastOp;
using CandidateTy =
    std::tuple<MaterializeOpTy, ReinterpretOpTy, Value /* funcArg*/>;
using CandidatesTy = SmallVector<CandidateTy>;

struct HexagonRVOPass : public ::impl::HexagonRVOBase<HexagonRVOPass> {

  void runOnOperation() override;
};

/// Replace linalg.generic outs that have no user with tensor.empty.
/// That simplifies analysis which try to determine if outs could be
/// replaced by the target of the `materialize in destionation` op.
void replaceAllocsWithEmpty(FunctionOpInterface funcOp, IRRewriter &rewriter) {
  funcOp.walk([&](linalg::GenericOp genericOp) {
    for (unsigned i = 0; i < genericOp.getNumDpsInits(); ++i) {
      OpOperand *operand = genericOp.getDpsInitOperand(i);
      if (!genericOp.getMatchingBlockArgument(operand).use_empty())
        continue;
      Value val = operand->get();
      auto TType = dyn_cast<TensorType>(val.getType());
      if (!TType)
        continue;

      auto elType = TType.getElementType();
      auto loc = genericOp.getLoc();
      rewriter.setInsertionPointAfterValue(val);
      auto empty = rewriter.create<tensor::EmptyOp>(
          loc, tensor::getMixedSizes(rewriter, loc, val), elType);
      genericOp.setDpsInitOperand(i, empty.getResult());
    }
  });
}

/// Use alias-analysis framework to work out if a particular memref
/// value is a view of the kernel input memref arg. The check below
/// is not the most/ eficient and when reverse resolve is fixed,
/// this can be reworked.
static std::pair<bool, Value>
isaViewOfFuncInputBuffer(Value value, FunctionOpInterface funcOp) {
  BufferViewFlowAnalysis analysis(funcOp);
  for (const auto &arg : funcOp.getArguments()) {
    if (isa<BaseMemRefType>(arg.getType())) {
      auto viewSet = analysis.resolve(arg);
      for (auto view : viewSet) {
        if (value == view)
          return {true, arg};
      }
    }
  }
  return {false, nullptr};
}

// A `materializeOp` is a candidate iff target is a view of
// func-arg memref and its singular view.
void populateCandidate(MaterializeOpTy op, FunctionOpInterface funcOp,
                       CandidatesTy &candidates) {
  auto castOp = op.getDest().getDefiningOp<ReinterpretOpTy>();
  if (!castOp)
    return;
  auto memref = castOp.getSource();
  auto [isViewOf, funcArg] = isaViewOfFuncInputBuffer(memref, funcOp);
  if (!isViewOf || !funcArg.hasOneUse())
    return;
  candidates.emplace_back(op, castOp, funcArg);
}

/// Return true if `neededValues` dominate (i.e. are available at)
/// insertion point.
bool doValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                    Operation *insertionPoint,
                                    const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      Block *owner = bbArg.getOwner();
      // block arg always dominates, provided IP is in` ownr`.
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = cast<OpResult>(val);
      if (!domInfo.properlyDominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

bool insertionPointDominatesUses(const DominanceInfo &domInfo,
                                 Operation *insertionPoint, Operation *op) {
  return llvm::all_of(op->getUsers(), [&](Operation *user) {
    return domInfo.dominates(insertionPoint, user);
  });
}

/// Find a new 'insertion point' for 'cast' that is as early
/// as possible i.e. when all needed values are available.
Operation *findEarlyInsertionPoint(CandidateTy &candidate) {
  DominanceInfo domInfo;
  SmallVector<Operation *> insertionPointCandidates;

  // function entry is first possibility.
  auto bbArg = cast<BlockArgument>(std::get<2>(candidate));
  insertionPointCandidates.push_back(
      &bbArg.getOwner()->getOperations().front());

  SmallVector<Value> neededValues;
  auto castOp = std::get<1>(candidate);
  ValueRange offsets = castOp.getOffsets();
  ValueRange sizes = castOp.getSizes();
  ValueRange strides = castOp.getStrides();

  neededValues.append(offsets.begin(), offsets.end());
  neededValues.append(sizes.begin(), sizes.end());
  neededValues.append(strides.begin(), strides.end());

  // Good early insertion points are:
  // a. front of block if needed value is block-arg; or
  // b. right after defining op of needed value
  for (auto val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      insertionPointCandidates.push_back(
          &bbArg.getOwner()->getOperations().front());
    } else {
      insertionPointCandidates.push_back(val.getDefiningOp()->getNextNode());
    }
  }

  // From all potential insertion points, pick one that
  // a. dominates all needed values (for correctness); and
  // b. is before the cast op
  for (Operation *insertionPoint : insertionPointCandidates) {
    if (!doValuesDominateInsertionPoint(domInfo, insertionPoint, neededValues))
      continue;
    if (!insertionPointDominatesUses(domInfo, insertionPoint, castOp))
      continue;
    return insertionPoint;
  }
  // No suitable insertion point was found.
  return nullptr;
}

/// Relocate the reinterpret-cast-op to 'as early as possible'
/// so that it is visible to bufferizer and eliminate empty tensor.
void replaceReinterpretOp(CandidatesTy &candidates, IRRewriter &rewriter) {
  for (auto &candidate : candidates) {
    auto materializeOp = std::get<0>(candidate);
    auto castOp = std::get<1>(candidate);
    DBG("current candidate: " << materializeOp);
    DBG(" -> dst castOp : " << castOp);

    auto insertionPoint = findEarlyInsertionPoint(candidate);
    if (!insertionPoint) {
      DBG(" -> copy elision not possible for : " << std::get<0>(candidate));
      continue;
    }
    rewriter.setInsertionPoint(insertionPoint);
    DBG(" -> insertion point : " << *insertionPoint);

    ValueRange offsets = castOp.getOffsets();
    ValueRange sizes = castOp.getSizes();
    ValueRange strides = castOp.getStrides();

    auto soffsets = castOp.getStaticOffsets();
    auto ssizes = castOp.getStaticSizes();
    auto sstrides = castOp.getStaticStrides();

    auto replacement = rewriter.create<ReinterpretOpTy>(
        castOp.getLoc(), castOp.getType(), castOp.getSource(), offsets, sizes,
        strides, soffsets, ssizes, sstrides);

    rewriter.replaceAllUsesWith(castOp.getResult(), replacement.getResult());
    materializeOp.setRestrict(true);
    DBG(" -> copy elision succeeded! replacement:" << replacement);
  }
}

void HexagonRVOPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  CandidatesTy candidates;

  replaceAllocsWithEmpty(funcOp, rewriter);
  funcOp.walk([&](MaterializeOpTy op) {
    populateCandidate(op, funcOp, candidates);
    return WalkResult::advance();
  });
  replaceReinterpretOp(candidates, rewriter);
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonRVOPass() {
  return std::make_unique<HexagonRVOPass>();
}
