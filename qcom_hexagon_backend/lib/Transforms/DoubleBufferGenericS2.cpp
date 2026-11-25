//===- DoubleBufferGenericS2.cpp - double buffer stage 2 implementation ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass implements stage 2 of the double buffering in Hexagon-MLIR.
// It takes the double-buffered IR and replaces the `memref.copy`s  with
// `memref.dma_start` and `memref.dma_wait` at the appropriate insertion points.
// See DoubleBufferGenericS1.cpp for details on stage-1.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Common/DMATransferUtil.h"
#include "hexagon/Common/Utils.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "double-buffer-generic-s2"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERGENERICS2
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// To parse the state of double-buffered graph IR.
struct KernelSchedule {
  scf::IfOp prefetch;
  Operation *compute;
  SmallVector<memref::CopyOp> stores;
};
struct DBSchedule {
  scf::IfOp prologue;
  scf::IfOp pongPrologue; // optional deep prologue for pong buffers
  scf::ForOp kernel;
  scf::IfOp pingKernel, pongKernel;
  KernelSchedule pingSchedule, pongSchedule;
};

/// Insert dma_waits using provided tags and num-elements.
void insertDMAWaits(Location loc, IRRewriter &rewriter, ArrayRef<Value> tags,
                    ArrayRef<Value> tagIndex, ArrayRef<Value> numElements) {
  for (int i = 0; i < tags.size(); ++i) {
    memref::DmaWaitOp::create(rewriter, loc, tags[i], tagIndex, numElements[i]);
  }
}

/// Replace `memref.copy` with `dma_start` using provided tags.
void replacePrefetchWithDMA(Location loc, IRRewriter &rewriter,
                            scf::IfOp schedule, ArrayRef<Value> tags) {
  SmallVector<memref::CopyOp, 3> copies;
  Region &thenRegion = schedule.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  Block &block = thenRegion.front();

  // warning: Don't be tempted to do replaceOp within this loop.
  for (Operation &op : block) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(&op))
      copies.push_back(copyOp);
  }

  for (int i = 0; i < copies.size(); ++i) {
    auto copy = copies[i];
    rewriter.setInsertionPoint(copy);
    bool created = createDMAStartOp(loc, rewriter, copy.getSource(),
                                    copy.getTarget(), tags[i]);
    assert(created && "unable to create dma_start");
    rewriter.eraseOp(copy);
  }
}

// Replace Stores with dma_start only.  The matching dma_wait is inserted at
// the start of the *opposite* kernel so the store DMA overlaps with the next
// tile's compute (see rewriteAsDMATransfers for placement details).
void replaceStoreshWithDMA(Location loc, IRRewriter &rewriter,
                           KernelSchedule &schedule, ArrayRef<Value> tags) {
  for (int i = 0; i < (int)schedule.stores.size(); ++i) {
    auto copy = schedule.stores[i];
    rewriter.setInsertionPoint(copy);
    bool created = createDMAStartOp(loc, rewriter, copy.getSource(),
                                    copy.getTarget(), tags[i]);
    assert(created && "unable to create dma_start for store back");
    rewriter.eraseOp(schedule.stores[i]);
  }
}

/// Rewrite the double-buffered loop operating on memref.copy as
/// dma_start and dma_wait sequence.
void rewriteAsDMATransfers(IRRewriter &rewriter, DBSchedule schedule) {
  RewriterBase::InsertionGuard guard(rewriter);
  Location loc = schedule.kernel.getLoc();

  // Tabulate preloads. A memref::StoreOp marks the boundary after which copies
  // belong to the inlined pong prologue (canonicalizer folded its scf.if).
  SmallVector<memref::CopyOp, 3> preloads;
  SmallVector<memref::CopyOp, 3> inlinedPongPrologueCopies;
  Region &thenRegion = schedule.prologue.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  Block &block = thenRegion.front();
  {
    bool pastStore = false;
    for (Operation &op : block) {
      if (isa<memref::StoreOp>(op)) {
        pastStore = true;
        continue;
      }
      if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        if (!pastStore)
          preloads.push_back(copyOp);
        else if (!schedule.pongPrologue)
          inlinedPongPrologueCopies.push_back(copyOp);
      }
    }
  }

  // Create tag-index
  rewriter.setInsertionPoint(schedule.prologue);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> tagIndex = {zero};

  // Generate the num-elements.
  SmallVector<Value, 3> numElements;
  SmallVector<Value, 3> numElementsStores;
  rewriter.setInsertionPoint(schedule.prologue);
  for (auto i = 0; i < preloads.size(); i++) {
    Value numEl = createNumElements(loc, rewriter, preloads[i].getSource());
    numElements.push_back(numEl);
  }

  // Generate num-elements for stores.
  // Ping and pongs are auto-gen'd and identical.
  auto numStores = schedule.pingSchedule.stores.size();
  for (auto i = 0; i < numStores; ++i) {
    Value numEl = createNumElements(
        loc, rewriter, schedule.pingSchedule.stores[i].getSource());
    numElementsStores.push_back(numEl);
  }

  // Allocate tags on the stack — they are tiny (4 bytes each) and short-lived;
  // alloca avoids heap malloc/free overhead on every kernel invocation.
  SmallVector<Value> pingWaits;
  SmallVector<Value> pongWaits;
  SmallVector<Value> pingStoreWaits;
  SmallVector<Value> pongStoreWaits;
  auto tagType = rewriter.getI32Type();
  auto waitMemrefType = MemRefType::get({1}, tagType);
  for (auto i = 0; i < preloads.size(); i++) {
    Value pingWait = memref::AllocaOp::create(rewriter, loc, waitMemrefType);
    Value pongWait = memref::AllocaOp::create(rewriter, loc, waitMemrefType);
    pingWaits.push_back(pingWait);
    pongWaits.push_back(pongWait);
  }
  for (auto i = 0; i < numStores; ++i) {
    Value ping = memref::AllocaOp::create(rewriter, loc, waitMemrefType);
    Value pong = memref::AllocaOp::create(rewriter, loc, waitMemrefType);
    pingStoreWaits.push_back(ping);
    pongStoreWaits.push_back(pong);
  }

  // (No deallocs needed — all tags are stack-allocated via alloca.)

  // hasRun flags guard each kernel's deferred store-wait, which is placed
  // just before the same kernel reuses its store tag on the next iteration.
  auto i1Type = rewriter.getI1Type();
  auto boolMemrefType = MemRefType::get({}, i1Type);
  Value pongHasRun = memref::AllocaOp::create(rewriter, loc, boolMemrefType);
  Value pingHasRun = memref::AllocaOp::create(rewriter, loc, boolMemrefType);
  Value falseConst =
      arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
  memref::StoreOp::create(rewriter, loc, falseConst, pongHasRun, ValueRange{});
  memref::StoreOp::create(rewriter, loc, falseConst, pingHasRun, ValueRange{});

  if (numStores > 0) {
    Value trueConst =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));

    // Ping deferred store-wait.
    rewriter.setInsertionPoint(schedule.pingSchedule.stores.front());
    {
      Value flag =
          memref::LoadOp::create(rewriter, loc, pingHasRun, ValueRange{});
      auto guardIf = scf::IfOp::create(rewriter, loc, flag, /*hasElse=*/false);
      RewriterBase::InsertionGuard ig(rewriter);
      rewriter.setInsertionPointToStart(guardIf.thenBlock());
      insertDMAWaits(loc, rewriter, pingStoreWaits, tagIndex,
                     numElementsStores);
    }

    // Pong deferred store-wait.
    rewriter.setInsertionPoint(schedule.pongSchedule.stores.front());
    {
      Value flag =
          memref::LoadOp::create(rewriter, loc, pongHasRun, ValueRange{});
      auto guardIf = scf::IfOp::create(rewriter, loc, flag, /*hasElse=*/false);
      RewriterBase::InsertionGuard ig(rewriter);
      rewriter.setInsertionPointToStart(guardIf.thenBlock());
      insertDMAWaits(loc, rewriter, pongStoreWaits, tagIndex,
                     numElementsStores);
    }

    // Set hasRun after each kernel's body for next iter's deferred wait.
    Block &pingBody = schedule.pingKernel.getThenRegion().front();
    rewriter.setInsertionPoint(pingBody.getTerminator());
    memref::StoreOp::create(rewriter, loc, trueConst, pingHasRun, ValueRange{});

    Block &pongBody = schedule.pongKernel.getThenRegion().front();
    rewriter.setInsertionPoint(pongBody.getTerminator());
    memref::StoreOp::create(rewriter, loc, trueConst, pongHasRun, ValueRange{});
  }

  // Epilogue: after the loop finishes, wait for the last outstanding store DMA.
  // The ping/pong flag reflects which kernel ran last:
  //   flag == false  =>  last was ping  =>  wait pingStoreWaits
  //   flag == true && pong_has_run  =>  last was pong  =>  wait pongStoreWaits
  //   flag == true && !pong_has_run =>  nothing ran   =>  no wait needed
  rewriter.setInsertionPointAfter(schedule.kernel);
  if (numStores > 0) {
    Value pingCond = schedule.pingKernel.getCondition();
    if (auto flagLoad = pingCond.getDefiningOp<memref::LoadOp>()) {
      Value flagAlloc = flagLoad.getMemRef();
      Value lastFlag =
          memref::LoadOp::create(rewriter, loc, flagAlloc, ValueRange{});
      Value phrFinal =
          memref::LoadOp::create(rewriter, loc, pongHasRun, ValueRange{});
      Value trueEpi =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      // flag == false => XOR true => true  (last kernel was ping)
      Value lastWasPing =
          arith::XOrIOp::create(rewriter, loc, lastFlag, trueEpi);
      // flag == true AND pong_has_run  (last kernel was pong)
      Value lastWasPong =
          arith::AndIOp::create(rewriter, loc, lastFlag, phrFinal);

      auto epiPingIf =
          scf::IfOp::create(rewriter, loc, lastWasPing, /*hasElse=*/false);
      {
        RewriterBase::InsertionGuard ig(rewriter);
        rewriter.setInsertionPointToStart(epiPingIf.thenBlock());
        insertDMAWaits(loc, rewriter, pingStoreWaits, tagIndex,
                       numElementsStores);
      }
      rewriter.setInsertionPointAfter(epiPingIf);

      auto epiPongIf =
          scf::IfOp::create(rewriter, loc, lastWasPong, /*hasElse=*/false);
      {
        RewriterBase::InsertionGuard ig(rewriter);
        rewriter.setInsertionPointToStart(epiPongIf.thenBlock());
        insertDMAWaits(loc, rewriter, pongStoreWaits, tagIndex,
                       numElementsStores);
      }
      rewriter.setInsertionPointAfter(epiPongIf);
    }
  }

  // Rewrite preloads to dma-starts.
  for (auto i = 0; i < preloads.size(); i++) {
    memref::CopyOp op = preloads[i];
    rewriter.setInsertionPoint(op);
    bool created = createDMAStartOp(loc, rewriter, op.getSource(),
                                    op.getTarget(), pingWaits[i]);
    assert(created && "unable to create dma_start");
    rewriter.eraseOp(op);
  }

  // Pong prologue copies → dma_start with pongWaits (drained by pong's first
  // dma_wait).
  if (schedule.pongPrologue) {
    SmallVector<memref::CopyOp, 3> pongPreloads;
    for (Operation &op : schedule.pongPrologue.getThenRegion().front())
      if (auto c = dyn_cast<memref::CopyOp>(op))
        pongPreloads.push_back(c);
    for (int i = 0; i < (int)pongPreloads.size(); ++i) {
      auto copy = pongPreloads[i];
      rewriter.setInsertionPoint(copy);
      bool ok = createDMAStartOp(loc, rewriter, copy.getSource(),
                                 copy.getTarget(), pongWaits[i]);
      assert(ok && "unable to create dma_start for pong prologue");
      rewriter.eraseOp(copy);
    }
  }

  // Same, for pong prologue copies that the canonicalizer inlined.
  for (int i = 0; i < (int)inlinedPongPrologueCopies.size(); ++i) {
    auto copy = inlinedPongPrologueCopies[i];
    rewriter.setInsertionPoint(copy);
    bool ok = createDMAStartOp(loc, rewriter, copy.getSource(),
                               copy.getTarget(), pongWaits[i]);
    assert(ok && "unable to create dma_start for inlined pong prologue");
    rewriter.eraseOp(copy);
  }

  // Insert ping and pong waits.
  rewriter.setInsertionPointAfter(schedule.pingSchedule.prefetch);
  insertDMAWaits(loc, rewriter, pingWaits, tagIndex, numElements);
  rewriter.setInsertionPointAfter(schedule.pongSchedule.prefetch);
  insertDMAWaits(loc, rewriter, pongWaits, tagIndex, numElements);

  // Replace prefetch copies with dma-starts.
  replacePrefetchWithDMA(loc, rewriter, schedule.pingSchedule.prefetch,
                         pongWaits);
  replacePrefetchWithDMA(loc, rewriter, schedule.pongSchedule.prefetch,
                         pingWaits);

  // Replace store copies with dma-starts; the matching dma-waits are
  // already placed at the start of the opposite kernel above.
  replaceStoreshWithDMA(loc, rewriter, schedule.pingSchedule, pingStoreWaits);
  replaceStoreshWithDMA(loc, rewriter, schedule.pongSchedule, pongStoreWaits);
}

/// Extract the prefetch section of ping-pong sub-kernels.
bool extractPreFetch(scf::IfOp kernel, KernelSchedule &schedule) {
  Region &thenRegion = kernel.getThenRegion();
  if (thenRegion.empty() || thenRegion.getBlocks().size() > 1)
    return false;
  Block &block = thenRegion.front();
  if (block.empty())
    return false;
  // Scan (not first-op): deep prologue may prepend scalar book-keeping.
  for (Operation &op : block) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (ifOp && ifOp->hasAttr("db_prefetch")) {
      schedule.prefetch = ifOp;
      return true;
    }
  }
  return false;
}

// Extract the store-back after kernel (ping or pong) compute.
bool extractStoreBack(scf::IfOp kernel, KernelSchedule &schedule) {
  // schedule.prefetch is already set by extractPreFetch
  if (!schedule.prefetch)
    return false;

  Operation *compute = schedule.prefetch->getNextNode();
  if (!compute)
    return false;

  // Skip the suppression-flag clearing store between prefetch and compute.
  while (compute && isa<memref::StoreOp>(compute))
    compute = compute->getNextNode();
  if (!compute)
    return false;

  if (!isa<scf::ForOp>(compute) && !isa<scf::ForallOp>(compute))
    return false;
  schedule.compute = compute;

  // After compute, collect VTCM→DDR memref::CopyOp till the yield.
  // Skip VTCM→VTCM copies (e.g. intermediate scratch-to-temp copies);
  // those are not valid DMA paths and must remain as synchronous copies.
  Operation *currentOp = compute->getNextNode();
  while (currentOp) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(currentOp))
      if (isVTCMtoDDR(copyOp))
        schedule.stores.push_back(copyOp);
    currentOp = currentOp->getNextNode();
  }
  if (schedule.stores.size() == 0)
    return false;
  return true;
}

/// State machine to parse the double buffered graph.
bool extractSchedule(scf::ForOp forOp, DBSchedule &schedule) {
  auto attr = forOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
  if (!attr)
    return false;
  int64_t id = attr.getInt();
  schedule.kernel = forOp;

  // Extract the prologue.
  bool foundPrologue = false;
  for (Operation *prev = forOp->getPrevNode(); prev != nullptr;
       prev = prev->getPrevNode()) {
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp)
      continue;
    auto attr = ifOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
    if (attr && id == attr.getInt() && ifOp->hasAttr("db_prologue")) {
      foundPrologue = true;
      schedule.prologue = ifOp;
      break;
    }
  }
  if (!foundPrologue)
    return false;
  DBG("Prologue = " << schedule.prologue);

  // Look for the optional deep pong prologue nested inside the prologue block.
  for (Operation &op : schedule.prologue.getThenRegion().front()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifOp->hasAttr("db_prologue_pong")) {
        schedule.pongPrologue = ifOp;
        break;
      }
    }
  }

  // Extract ping-pong kernels.
  bool pingKernelFound = false, pongKernelFound = false;
  for (Operation &op : *forOp.getBody()) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    if (auto attr = ifOp->hasAttr("db_ping_kernel")) {
      assert(!pingKernelFound && !pongKernelFound &&
             "multiple ping kernel not expected");
      schedule.pingKernel = ifOp;
      pingKernelFound = true;
      continue;
    }
    if (auto attr = ifOp->hasAttr("db_pong_kernel")) {
      assert(pingKernelFound && !pongKernelFound &&
             "reverse  ping pong not expected");
      schedule.pongKernel = ifOp;
      pongKernelFound = true;
      break;
    }
  }
  if (!pingKernelFound || !pongKernelFound)
    return false;

  if (!extractPreFetch(schedule.pingKernel, schedule.pingSchedule) ||
      !extractPreFetch(schedule.pongKernel, schedule.pongSchedule))
    return false;

  if (!extractStoreBack(schedule.pingKernel, schedule.pingSchedule) ||
      !extractStoreBack(schedule.pongKernel, schedule.pongSchedule) ||
      schedule.pingSchedule.stores.size() !=
          schedule.pongSchedule.stores.size())
    return false;
  return true;
}

// The concrete implementation of the Double-Buffer-Generic-Stage-2 pass.
struct HexagonDoubleBufferGenericS2Pass
    : public ::impl::HexagonDoubleBufferGenericS2Base<
          HexagonDoubleBufferGenericS2Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    func.walk([](scf::ForOp forOp) {
      DBSchedule schedule;
      bool res = extractSchedule(forOp, schedule);
      if (res) {
        IRRewriter rewriter(forOp.getContext());
        rewriteAsDMATransfers(rewriter, schedule);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Creates an instance of the DoubleBufferGenericS2 pass.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS2Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS2Pass>();
}
