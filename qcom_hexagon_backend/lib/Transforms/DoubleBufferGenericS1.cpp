//===- DoubleBufferGenericS1.cpp - Double Buffer Generic Pass : Stage 1 ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass transforms single-buffered tiled linalg-generic loops into
// software-pipelined double-buffered loops using ping-pong buffers to overlap
// DMA transfers with computation. It allocates two buffer sets, creates a
// prologue to prefetch the first iteration, and generates alternating ping/pong
// sub-kernels that prefetch the next iteration while computing the current one.
//
// This implementation is mainly in two main passes. First is this one and
// the next is in DoubleBufferGenericS2.cpp
//
// Future extensions will support additional cases including >2D DMA transfers,
// reduction loops, multi-buffering strategies, and broader pattern coverage.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/DMATransferUtil.h"
#include "hexagon/Common/Utils.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <optional>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "double-buffer-generic-s1"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERGENERICS1
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Extract a constant `index` value from an SSA `Value`.
static std::optional<int64_t> getIndexConstant(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (llvm::isa<mlir::IndexType>(cOp.getType())) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cOp.getValue()))
        return intAttr.getInt(); // signed 64-bit
    }
  }
  // Optional: peel a trivial index cast from an integer constant.
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    if (auto srcConst = cast.getIn().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(srcConst.getValue()))
        return intAttr.getInt();
    }
  }
  return std::nullopt;
}

// Returns true if `forOp` is in `normal form` with
// knowns bounds and deterministic size of fetches.
bool isZeroBasedEvenlyDivisible(scf::ForOp forOp) {
  auto lb = getIndexConstant(forOp.getLowerBound());
  auto ub = getIndexConstant(forOp.getUpperBound());
  auto st = getIndexConstant(forOp.getStep());
  if (!lb || !ub || !st || *lb != 0 || *st == 0 || *ub % *st != 0)
    return false;
  return true;
}

/// Structs to parse and store the schedule.
enum class ScheduleStage {
  View,
  Alloc,
  Copy,
  Compute,
  InlineCompute,
  Store,
  Unknown
};
struct ScheduleTriplet {
  memref::SubViewOp view;
  memref::AllocOp alloc;
  memref::CopyOp copy;
  bool writeOnly = false; // for write-only slices.
};

struct SingleBufferSchedule {
  scf::ForOp forOp;
  SmallVector<ScheduleTriplet> triplets;
  // Compute section: either a single ForallOp/tagged-ForOp (legacy path) or a
  // sequence of inline ops (PromoteTiledSCFLoopToVTCM path tagged all_parallel
  // + tiled_generic on the outer loop).
  SmallVector<Operation *> computeOps;
  SmallVector<memref::SubViewOp> storeViews; // output subviews into the store.
  SmallVector<memref::CopyOp> stores;
};

/// Utility to convert `value` to `index` type.
Value convertToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  auto indexType = builder.getIndexType();
  auto indexValue =
      mlir::arith::IndexCastOp::create(builder, loc, indexType, value);
  return indexValue;
}

// Generate the ping or pong region IR of the double-buffered loop.
// `pongPrologueDone` (ping only): suppresses iter 0's pong prefetch when the
// deep prologue has already loaded the first pong tile.
void generatePingPongSubKernel(
    IRRewriter &rewriter, SingleBufferSchedule &schedule,
    scf::LoopNest &dbLoopNest, scf::IfOp ppongIfOp, Value nextExists,
    Value nextIdx, ArrayRef<Value> thisBuffers, ArrayRef<Value> nextBuffers,
    Value toggle, Value toggleNextStoreValue, Value pongPrologueDone = nullptr,
    Value falseVal = nullptr) {
  auto forOp = schedule.forOp;
  auto loc = forOp.getLoc();
  auto context = forOp.getContext();
  auto indVar = forOp.getInductionVar();

  SmallVector<Value> ivs = llvm::map_to_vector(
      dbLoopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  assert(ivs.size() == 1 && "expecting a single loop at this point");
  Block *forBody = dbLoopNest.loops.back().getBody();
  Value newDBLoopIndVar = dbLoopNest.loops.back().getInductionVar();

  // Create prefetch section: executes if this is not the last iteration.
  rewriter.setInsertionPointToStart(&ppongIfOp.getThenRegion().front());
  Value doPrefetch = nextExists;
  if (pongPrologueDone) {
    Value trueVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    Value done = memref::LoadOp::create(rewriter, loc, rewriter.getI1Type(),
                                        pongPrologueDone);
    Value notDone = arith::XOrIOp::create(rewriter, loc, done, trueVal);
    doPrefetch = arith::AndIOp::create(rewriter, loc, nextExists, notDone);
  }
  auto ifNotLastIter =
      scf::IfOp::create(rewriter, loc, TypeRange(), doPrefetch, false);
  ifNotLastIter->setAttr("db_prefetch", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifNotLastIter.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    if (schedule.triplets[i].writeOnly)
      continue; // write-only outputs have no prefetch copy
    IRMapping mapping;
    mapping.map(indVar, nextIdx);
    auto view = schedule.triplets[i].view;
    Operation *newViewOp = view->clone(mapping);
    rewriter.insert(newViewOp);
    Value cloneResult = newViewOp->getResult(0);
    memref::CopyOp::create(rewriter, loc, cloneResult, nextBuffers[i]);
  }

  // Re-map the compute.
  rewriter.setInsertionPointAfter(ifNotLastIter);
  // Clear suppression flag for iter 1+.
  if (pongPrologueDone)
    memref::StoreOp::create(rewriter, loc, falseVal, pongPrologueDone,
                            ValueRange{});
  IRMapping mapping2;
  mapping2.map(indVar, newDBLoopIndVar);
  for (auto i = 0; i < schedule.triplets.size(); ++i)
    mapping2.map(schedule.triplets[i].alloc, thisBuffers[i]);
  for (Operation *computeOp : schedule.computeOps) {
    Operation *newComputeOp = computeOp->clone(mapping2);
    rewriter.insert(newComputeOp);
    for (auto [oldVal, newVal] :
         llvm::zip(computeOp->getResults(), newComputeOp->getResults()))
      mapping2.map(oldVal, newVal);
  }

  // Store the results.
  for (auto &storeView : schedule.storeViews) {
    IRMapping mapping;
    mapping.map(indVar, newDBLoopIndVar);
    Operation *newViewOp = storeView->clone(mapping);
    rewriter.insert(newViewOp);
    mapping2.map(storeView.getResult(), newViewOp->getResult(0));
  }
  // Clone input triplet views (skip write-only) for stores.
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    if (schedule.triplets[i].writeOnly)
      continue;
    IRMapping mapping;
    mapping.map(indVar, newDBLoopIndVar);
    auto view = schedule.triplets[i].view;
    Operation *newViewOp = view->clone(mapping);
    rewriter.insert(newViewOp);
    Value cloneResult = newViewOp->getResult(0);
    mapping2.map(view, cloneResult);
  }
  for (auto i = 0; i < schedule.stores.size(); ++i) {
    Operation *newStore = schedule.stores[i]->clone(mapping2);
    rewriter.insert(newStore);
  }
  memref::StoreOp::create(rewriter, loc, toggleNextStoreValue, toggle,
                          ValueRange{});
}

/// Rewrite the single-buffered loop as double buffered.
void rewriteAsDoubleBuffered(IRRewriter &rewriter, scf::ForOp sbForOp,
                             SingleBufferSchedule &schedule, int &uid) {
  auto loc = sbForOp.getLoc();
  auto context = sbForOp.getContext();

  // Define some general types.
  auto boolType = rewriter.getI1Type();
  auto boolMemrefType = MemRefType::get({}, boolType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sbForOp);

  // Define some general constants.
  Value trueVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                            rewriter.getBoolAttr(true));
  Value falseVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                             rewriter.getBoolAttr(false));

  // Allocate `toggle` for ping-pong state and initialize it to true.
  Value toggle = memref::AllocOp::create(rewriter, loc, boolMemrefType);
  auto store =
      memref::StoreOp::create(rewriter, loc, trueVal, toggle, ValueRange{});

  // Allocate Ping-Pong Buffers.
  int64_t alignment = 2048;
  auto alignmentAttr = rewriter.getI64IntegerAttr(alignment);
  SmallVector<Value, 3> pingBuffers;
  SmallVector<Value, 3> pongBuffers;
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    auto alloc = schedule.triplets[i].alloc;
    Value ping =
        memref::AllocOp::create(rewriter, loc << "ping_buffer", alloc.getType(),
                                mlir::ValueRange{}, alignmentAttr);
    Value pong =
        memref::AllocOp::create(rewriter, loc << "pong_buffer", alloc.getType(),
                                mlir::ValueRange{}, alignmentAttr);
    pingBuffers.push_back(ping);
    pongBuffers.push_back(pong);
  }

  // Get the bounds from the single-buffer original for-loop.
  Value lowerBoundIdx = convertToIndex(rewriter, loc, sbForOp.getLowerBound());
  Value upperBoundIdx = convertToIndex(rewriter, loc, sbForOp.getUpperBound());
  Value stepIdx = convertToIndex(rewriter, loc, sbForOp.getStep());
  auto indVar = sbForOp.getInductionVar();
  auto idAttr = mlir::IntegerAttr::get(rewriter.getI64Type(), uid++);

  // Prologue: executes iff not 0-iteration loop.
  Value mayLoop =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            lowerBoundIdx, upperBoundIdx)
          .getResult();
  Value mayLoopVar = memref::AllocOp::create(rewriter, loc, boolMemrefType);
  memref::StoreOp::create(rewriter, loc, mayLoop, mayLoopVar, ValueRange{});
  Value mayLoopReFetch =
      memref::LoadOp::create(rewriter, loc, boolType, mayLoopVar);
  auto ifMayLoop =
      scf::IfOp::create(rewriter, loc, TypeRange(), mayLoopReFetch, false);
  ifMayLoop->setAttr("db_generic", idAttr);
  ifMayLoop->setAttr("db_prologue", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifMayLoop.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    if (schedule.triplets[i].writeOnly)
      continue; // write-only outputs need no prologue prefetch
    IRMapping mapping;
    mapping.map(indVar, lowerBoundIdx);
    auto view = schedule.triplets[i].view;
    Operation *newViewOp = view->clone(mapping);
    rewriter.insert(newViewOp);
    Value cloneResult = newViewOp->getResult(0);
    memref::CopyOp::create(rewriter, loc, cloneResult, pingBuffers[i]);
  }

  // Deep prologue: prefetch tile 1 into pong so pong iter 0 has input ready.
  // Suppression flag is allocated outside ifMayLoop so the main loop sees it.
  rewriter.setInsertionPoint(ifMayLoop);
  Value pongPrologueDone =
      memref::AllocOp::create(rewriter, loc, boolMemrefType);
  memref::StoreOp::create(rewriter, loc, falseVal, pongPrologueDone,
                          ValueRange{});

  rewriter.setInsertionPoint(ifMayLoop.getThenRegion().front().getTerminator());
  {
    Value tile1Idx =
        arith::AddIOp::create(rewriter, loc, lowerBoundIdx, stepIdx)
            .getResult();
    // Fold tile1Exists when bounds are static; canonicalizer folds the
    // resulting scf.if, and S2 picks up the inlined copies.
    Value tile1Exists;
    auto lowerCst = getConstantIntValue(lowerBoundIdx);
    auto upperCst = getConstantIntValue(upperBoundIdx);
    auto stepCst = getConstantIntValue(stepIdx);
    if (lowerCst && upperCst && stepCst)
      tile1Exists = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getBoolAttr((*lowerCst + *stepCst) < *upperCst));
    else
      tile1Exists =
          arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                                tile1Idx, upperBoundIdx)
              .getResult();
    // Record so ping iter 0 can skip its pong prefetch.
    memref::StoreOp::create(rewriter, loc, tile1Exists, pongPrologueDone,
                            ValueRange{});
    auto ifTile1 =
        scf::IfOp::create(rewriter, loc, TypeRange(), tile1Exists, false);
    ifTile1->setAttr("db_prologue_pong", UnitAttr::get(context));
    rewriter.setInsertionPointToStart(&ifTile1.getThenRegion().front());
    for (auto i = 0; i < schedule.triplets.size(); ++i) {
      if (schedule.triplets[i].writeOnly)
        continue;
      IRMapping mapping;
      mapping.map(indVar, tile1Idx);
      auto view = schedule.triplets[i].view;
      Operation *newViewOp = view->clone(mapping);
      rewriter.insert(newViewOp);
      Value cloneResult = newViewOp->getResult(0);
      memref::CopyOp::create(rewriter, loc, cloneResult, pongBuffers[i]);
    }
  }

  // Kernel: create the new double-buffered top-loop.
  rewriter.setInsertionPoint(sbForOp);
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, SmallVector<Value>{lowerBoundIdx},
      SmallVector<Value>{upperBoundIdx}, SmallVector<Value>{stepIdx});
  loopNest.loops.back()->setAttr("db_generic", idAttr);
  Block *forBody = loopNest.loops.back().getBody();
  Value dbIndVar = loopNest.loops.back().getInductionVar();

  // Toggle decides whether this is ping-or-pong-stage.
  rewriter.setInsertionPoint(forBody->getTerminator());
  Value toggleVal = memref::LoadOp::create(rewriter, loc, boolType, toggle);

  // Check if next preload should happen (or this is last iteration).
  Value nextIdx =
      arith::AddIOp::create(rewriter, loc, dbIndVar, stepIdx).getResult();
  Value nextExists =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            nextIdx, upperBoundIdx)
          .getResult();

  // Ping sub-kernel: pass pongPrologueDone so iter 0 skips its pong prefetch.
  auto pingIfOp =
      scf::IfOp::create(rewriter, loc, TypeRange(), toggleVal, false);
  pingIfOp->setAttr("db_ping_kernel", UnitAttr::get(context));
  generatePingPongSubKernel(rewriter, schedule, loopNest, pingIfOp, nextExists,
                            nextIdx, pingBuffers, pongBuffers, toggle,
                            falseVal /*toggleNextStoreValue*/, pongPrologueDone,
                            falseVal);

  // Pong sub-kernel.
  rewriter.setInsertionPointAfter(pingIfOp);
  Value invertedToggleVal =
      arith::XOrIOp::create(rewriter, loc, toggleVal, trueVal);
  auto pongIfOp =
      scf::IfOp::create(rewriter, loc, TypeRange(), invertedToggleVal, false);
  pongIfOp->setAttr("db_pong_kernel", UnitAttr::get(context));
  generatePingPongSubKernel(rewriter, schedule, loopNest, pongIfOp, nextExists,
                            nextIdx, pongBuffers, pingBuffers, toggle,
                            trueVal /*toggleNextStoreValue*/);
  rewriter.eraseOp(sbForOp);
}

// State machine to parse the `tiled_generic`.
bool generateSchedule(IRRewriter &rewriter, scf::ForOp forOp,
                      SingleBufferSchedule &schedule) {
  if (!forOp.getInitArgs().empty())
    return false;
  schedule.forOp = forOp;

  bool verifiedTiledGenericParallel = false;
  if (forOp->hasAttr("tiled_generic") && forOp->hasAttr("all_parallel"))
    verifiedTiledGenericParallel = true;

  // For loops tagged by PromoteTiledSCFLoopToVTCM we trust the tile structure
  // and skip the zero-based / evenly-divisible guard (the loop may start at a
  // non-zero grid offset and have runtime bounds).  For all other loops keep
  // the existing guard to avoid false matches.
  if (!verifiedTiledGenericParallel && !isZeroBasedEvenlyDivisible(forOp))
    return false;

  Block *forBody = forOp.getBody();
  ScheduleTriplet triplet;
  ScheduleStage stage = ScheduleStage::View;
  for (size_t i = 0; i < forBody->getOperations().size(); i++) {
    Operation &op = *std::next(forBody->begin(), i);
    switch (stage) {
    case ScheduleStage::View: {
      auto viewOp = llvm::dyn_cast<memref::SubViewOp>(op);
      if (!viewOp) {
        // A standalone AllocOp (without a preceding subview/copy) is a
        // write-only output: it gets VTCM space but no DDR->VTCM prefetch.
        if (auto allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
          ScheduleTriplet woTriplet;
          woTriplet.alloc = allocOp;
          woTriplet.writeOnly = true;
          schedule.triplets.push_back(woTriplet);
          break; // stay in View state, more triplets or compute may follow
        }
        // Skip preamble ops (e.g. index_cast of the loop IV) that appear
        // before the first SubView in tagged loops.
        if (schedule.triplets.empty()) {
          if (verifiedTiledGenericParallel)
            break; // skip silently and stay in View
          return false;
        }
        // After at least one triplet: a ForallOp or a single tagged ForOp is
        // the (legacy) compute wrapper — go to Store expecting a direct copy.
        if (llvm::isa<scf::ForallOp>(op) ||
            (llvm::isa<scf::ForOp>(op) && verifiedTiledGenericParallel)) {
          schedule.computeOps.push_back(&op);
          stage = ScheduleStage::Store;
          break;
        }
        // For PromoteTiledSCFLoopToVTCM the compute is a sequence of inline ops
        // (linalg.generic, linalg.fill, allocs, copies, …).  Transition to
        // InlineCompute to collect them; first top-level SubViewOp = store.
        if (verifiedTiledGenericParallel && !llvm::isa<scf::YieldOp>(op)) {
          schedule.computeOps.push_back(&op);
          stage = ScheduleStage::InlineCompute;
          break;
        }
        return false;
      }
      triplet.view = viewOp;
      stage = ScheduleStage::Alloc;
      break;
    }
    case ScheduleStage::Alloc: {
      auto allocOp = llvm::dyn_cast<memref::AllocOp>(op);
      if (!allocOp)
        return false;
      triplet.alloc = allocOp;
      stage = ScheduleStage::Copy;
      break;
    }
    case ScheduleStage::Copy: {
      auto copyOp = llvm::dyn_cast<memref::CopyOp>(op);
      // S2 turns this copy into a DMA.
      if (!copyOp || !canCreateDMAStart(copyOp.getSource().getType(),
                                        copyOp.getTarget().getType()))
        return false;
      triplet.copy = copyOp;
      schedule.triplets.push_back(triplet);
      stage = ScheduleStage::View;
      break;
    }
    case ScheduleStage::InlineCompute: {
      // Collect inline compute ops until the first SubViewOp, which signals
      // the start of the store section.
      if (auto viewOp = llvm::dyn_cast<memref::SubViewOp>(op)) {
        schedule.storeViews.push_back(viewOp);
        stage = ScheduleStage::Store;
        break;
      }
      if (llvm::isa<scf::YieldOp>(op))
        return false; // no store section — not supported
      schedule.computeOps.push_back(&op);
      break;
    }
    case ScheduleStage::Store: {
      if (llvm::dyn_cast<scf::YieldOp>(op)) {
        if (schedule.stores.empty())
          return false;
        return true; // done!
      }
      // A SubViewOp in the store section is the DDR slice for a write-only.
      if (auto viewOp = llvm::dyn_cast<memref::SubViewOp>(op)) {
        schedule.storeViews.push_back(viewOp);
        break; // from switch
      }
      auto copyOp = llvm::dyn_cast<memref::CopyOp>(op);
      // TODO: more checks about what is being stored.
      if (!copyOp) {
        // For the vectorized PromoteTiledSCFLoopToVTCM path, the first ForOp in
        // View state took us to Store, but more compute ops follow before the
        // real store SubViewOp.  Step back into InlineCompute to collect them;
        // the next top-level SubViewOp will re-trigger the Store transition.
        if (verifiedTiledGenericParallel) {
          schedule.computeOps.push_back(&op);
          stage = ScheduleStage::InlineCompute;
          break;
        }
        return false;
      }
      // S2 turns VTCM->DDR stores into DMAs.
      if (isVTCMtoDDR(copyOp) &&
          !canCreateDMAStart(copyOp.getSource().getType(),
                             copyOp.getTarget().getType()))
        return false;
      schedule.stores.push_back(copyOp);
      break;
    }
    default:
      return false;
    } // switch
  }   // for
  return false;
}

// The concrete implementation of the DoubleBufferGenericS1 pass.
struct HexagonDoubleBufferGenericS1Pass
    : public ::impl::HexagonDoubleBufferGenericS1Base<
          HexagonDoubleBufferGenericS1Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    int uniqueID = 0; // for each double-buffered linalg-generic.
    auto func = getOperation();

    func.walk([&uniqueID](scf::ForOp forOp) {
      SingleBufferSchedule schedule;
      IRRewriter rewriter(forOp.getContext());
      bool viableCandidate = generateSchedule(rewriter, forOp, schedule);
      if (viableCandidate)
        rewriteAsDoubleBuffered(rewriter, forOp, schedule, uniqueID);
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Creates an instance of the DoubleBufferGenericS1 pass.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS1Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS1Pass>();
}
