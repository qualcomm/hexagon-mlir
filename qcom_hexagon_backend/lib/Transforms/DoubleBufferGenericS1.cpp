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

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
enum class ScheduleStage { View, Alloc, Copy, Compute, Store, Unknown };
struct ScheduleTriplet {
  memref::SubViewOp view;
  memref::AllocOp alloc;
  memref::CopyOp copy;
};

struct SingleBufferSchedule {
  scf::ForOp forOp;
  SmallVector<ScheduleTriplet> triplets;
  Operation *compute;
  SmallVector<memref::CopyOp> stores;
};

/// Utility to convert `value` to `index` type.
Value convertToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  auto indexType = builder.getIndexType();
  auto indexValue =
      builder.create<mlir::arith::IndexCastOp>(loc, indexType, value);
  return indexValue;
}

// Generate the ping or pong region IR of the double-buffered loop.
void generatePingPongSubKernel(IRRewriter &rewriter,
                               SingleBufferSchedule &schedule,
                               scf::LoopNest &dbLoopNest, scf::IfOp ppongIfOp,
                               Value nextExists, Value nextIdx,
                               ArrayRef<Value> thisBuffers,
                               ArrayRef<Value> nextBuffers, Value toggle,
                               Value toggleNextStoreValue) {
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
  auto ifNotLastIter =
      rewriter.create<scf::IfOp>(loc, TypeRange(), nextExists, false);
  ifNotLastIter->setAttr("db_prefetch", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifNotLastIter.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, nextIdx);
    auto view = schedule.triplets[i].view;
    Operation *newViewOp = view->clone(mapping);
    rewriter.insert(newViewOp);
    Value cloneResult = newViewOp->getResult(0);
    rewriter.create<memref::CopyOp>(loc, cloneResult, nextBuffers[i]);
  }

  // Re-map the compute.
  rewriter.setInsertionPointAfter(ifNotLastIter);
  IRMapping mapping2;
  mapping2.map(indVar, newDBLoopIndVar);
  for (auto i = 0; i < schedule.triplets.size(); ++i)
    mapping2.map(schedule.triplets[i].alloc, thisBuffers[i]);
  Operation *newComputeOp = schedule.compute->clone(mapping2);
  rewriter.insert(newComputeOp);

  // Store the results.
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
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
  rewriter.create<memref::StoreOp>(loc, toggleNextStoreValue, toggle,
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
  Value trueVal = rewriter.create<arith::ConstantOp>(
      loc, boolType, rewriter.getBoolAttr(true));
  Value falseVal = rewriter.create<arith::ConstantOp>(
      loc, boolType, rewriter.getBoolAttr(false));

  // Allocate `toggle` for ping-pong state and initialize it to true.
  Value toggle = rewriter.create<memref::AllocOp>(loc, boolMemrefType);
  auto store =
      rewriter.create<memref::StoreOp>(loc, trueVal, toggle, ValueRange{});

  // Allocate Ping-Pong Buffers.
  int64_t alignment = 2048;
  auto alignmentAttr = rewriter.getI64IntegerAttr(alignment);
  SmallVector<Value, 3> pingBuffers;
  SmallVector<Value, 3> pongBuffers;
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    auto alloc = schedule.triplets[i].alloc;
    Value ping = rewriter.create<memref::AllocOp>(
        loc, alloc.getType(), mlir::ValueRange{}, alignmentAttr);
    Value pong = rewriter.create<memref::AllocOp>(
        loc, alloc.getType(), mlir::ValueRange{}, alignmentAttr);
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
      rewriter
          .create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                 lowerBoundIdx, upperBoundIdx)
          .getResult();
  Value mayLoopVar = rewriter.create<memref::AllocOp>(loc, boolMemrefType);
  rewriter.create<memref::StoreOp>(loc, mayLoop, mayLoopVar, ValueRange{});
  Value mayLoopReFetch =
      rewriter.create<memref::LoadOp>(loc, boolType, mayLoopVar);
  auto ifMayLoop =
      rewriter.create<scf::IfOp>(loc, TypeRange(), mayLoopReFetch, false);
  ifMayLoop->setAttr("db_generic", idAttr);
  ifMayLoop->setAttr("db_prologue", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifMayLoop.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, lowerBoundIdx);
    auto view = schedule.triplets[i].view;
    Operation *newViewOp = view->clone(mapping);
    rewriter.insert(newViewOp);
    Value cloneResult = newViewOp->getResult(0);
    rewriter.create<memref::CopyOp>(loc, cloneResult, pingBuffers[i]);
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
  Value toggleVal = rewriter.create<memref::LoadOp>(loc, boolType, toggle);

  // Check if next preload should happen (or this is last iteration).
  Value nextIdx =
      rewriter.create<arith::AddIOp>(loc, dbIndVar, stepIdx).getResult();
  Value nextIdxPlusTileSize =
      rewriter.create<arith::AddIOp>(loc, nextIdx, stepIdx).getResult();
  Value nextExists =
      rewriter
          .create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle,
                                 nextIdxPlusTileSize, upperBoundIdx)
          .getResult();

  // Ping sub-kernel.
  auto pingIfOp =
      rewriter.create<scf::IfOp>(loc, TypeRange(), toggleVal, false);
  pingIfOp->setAttr("db_ping_kernel", UnitAttr::get(context));
  generatePingPongSubKernel(rewriter, schedule, loopNest, pingIfOp, nextExists,
                            nextIdx, pingBuffers, pongBuffers, toggle,
                            falseVal /*toggleNextStoreValue*/);

  // Pong sub-kernel.
  rewriter.setInsertionPointAfter(pingIfOp);
  Value invertedToggleVal =
      rewriter.create<arith::XOrIOp>(loc, toggleVal, trueVal);
  auto pongIfOp =
      rewriter.create<scf::IfOp>(loc, TypeRange(), invertedToggleVal, false);
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

  if (!isZeroBasedEvenlyDivisible(forOp))
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
        if (schedule.triplets.empty())
          return false;
        if (llvm::isa<scf::ForallOp>(op) ||
            (llvm::isa<scf::ForOp>(op) && verifiedTiledGenericParallel)) {
          schedule.compute = &op;
          stage = ScheduleStage::Store;
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
      if (!copyOp)
        return false;
      triplet.copy = copyOp;
      schedule.triplets.push_back(triplet);
      stage = ScheduleStage::View;
      break;
    }
    case ScheduleStage::Store: {
      if (llvm::dyn_cast<scf::YieldOp>(op)) {
        if (schedule.stores.empty())
          return false;
        return true; // done!
      }
      auto copyOp = llvm::dyn_cast<memref::CopyOp>(op);
      // TODO: more checks about what is being stored.
      if (!copyOp)
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
