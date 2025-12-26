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
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Conversion/Passes.h"
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
  scf::ForOp kernel;
  scf::IfOp pingKernel, pongKernel;
  KernelSchedule pingSchedule, pongSchedule;
};

/// Insert dma_waits using provided tags and num-elements.
void insertDMAWaits(Location loc, IRRewriter &rewriter, ArrayRef<Value> tags,
                    ArrayRef<Value> tagIndex, ArrayRef<Value> numElements) {
  for (int i = 0; i < tags.size(); ++i) {
    rewriter.create<memref::DmaWaitOp>(loc, tags[i], tagIndex, numElements[i]);
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

// Replace Stores with DMA Waits.
void replaceStoreshWithDMA(Location loc, IRRewriter &rewriter,
                           KernelSchedule &schedule, ArrayRef<Value> tags,
                           ArrayRef<Value> tagIndex,
                           ArrayRef<Value> numElements) {
  auto tagType = rewriter.getI32Type();
  auto waitMemrefType = MemRefType::get({1}, tagType);

  for (int i = 0; i < schedule.stores.size(); ++i) {
    auto copy = schedule.stores[i];
    rewriter.setInsertionPoint(copy);
    bool created = createDMAStartOp(loc, rewriter, copy.getSource(),
                                    copy.getTarget(), tags[i]);
    rewriter.create<memref::DmaWaitOp>(loc, tags[i], tagIndex, numElements[i]);
    rewriter.eraseOp(schedule.stores[i]);
  }
}

/// Rewrite the double-buffered loop operating on memref.copy as
/// dma_start and dma_wait sequence.
void rewriteAsDMATransfers(IRRewriter &rewriter, DBSchedule schedule) {
  RewriterBase::InsertionGuard guard(rewriter);
  Location loc = schedule.kernel.getLoc();

  // Tabulate preloads.
  SmallVector<memref::CopyOp, 3> preloads;
  Region &thenRegion = schedule.prologue.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  Block &block = thenRegion.front();
  for (Operation &op : block) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(&op))
      preloads.push_back(copyOp);
  }

  // Create tag-index
  rewriter.setInsertionPoint(schedule.prologue);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
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

  // Allocate tags.
  SmallVector<Value> pingWaits;
  SmallVector<Value> pongWaits;
  SmallVector<Value> pingStoreWaits;
  SmallVector<Value> pongStoreWaits;
  auto tagType = rewriter.getI32Type();
  auto waitMemrefType = MemRefType::get({1}, tagType);
  for (auto i = 0; i < preloads.size(); i++) {
    Value pingWait = rewriter.create<memref::AllocOp>(loc, waitMemrefType);
    Value pongWait = rewriter.create<memref::AllocOp>(loc, waitMemrefType);
    pingWaits.push_back(pingWait);
    pongWaits.push_back(pongWait);
  }
  for (auto i = 0; i < numStores; ++i) {
    Value ping = rewriter.create<memref::AllocOp>(loc, waitMemrefType);
    Value pong = rewriter.create<memref::AllocOp>(loc, waitMemrefType);
    pingStoreWaits.push_back(ping);
    pongStoreWaits.push_back(pong);
  }

  // Deallocte tags.
  rewriter.setInsertionPointAfter(schedule.kernel);
  for (auto i = 0; i < preloads.size(); i++) {
    rewriter.create<memref::DeallocOp>(loc, pingWaits[i]);
    rewriter.create<memref::DeallocOp>(loc, pongWaits[i]);
  }
  for (auto i = 0; i < numStores; ++i) {
    rewriter.create<memref::DeallocOp>(loc, pingStoreWaits[i]);
    rewriter.create<memref::DeallocOp>(loc, pongStoreWaits[i]);
  }

  // Rewrite preloads to dma-starts
  for (auto i = 0; i < preloads.size(); i++) {
    memref::CopyOp op = preloads[i];
    rewriter.setInsertionPoint(op);
    bool created = createDMAStartOp(loc, rewriter, op.getSource(),
                                    op.getTarget(), pingWaits[i]);
    assert(created && "unable to create dma_start");
    rewriter.eraseOp(op);
  }

  // Insert ping and pong waits.
  rewriter.setInsertionPointAfter(schedule.pingSchedule.prefetch);
  insertDMAWaits(loc, rewriter, pingWaits, tagIndex, numElements);
  rewriter.setInsertionPointAfter(schedule.pongSchedule.prefetch);
  insertDMAWaits(loc, rewriter, pongWaits, tagIndex, numElements);

  // Replace memref.copy with dma-start (and dma-wait for stores).
  replacePrefetchWithDMA(loc, rewriter, schedule.pingSchedule.prefetch,
                         pongWaits);
  replacePrefetchWithDMA(loc, rewriter, schedule.pongSchedule.prefetch,
                         pingWaits);

  replaceStoreshWithDMA(loc, rewriter, schedule.pingSchedule, pingStoreWaits,
                        tagIndex, numElementsStores);
  replaceStoreshWithDMA(loc, rewriter, schedule.pongSchedule, pongStoreWaits,
                        tagIndex, numElementsStores);
}

/// Extract the prefetch section of ping-pong sub-kernels.
bool extractPreFetch(scf::IfOp kernel, KernelSchedule &schedule) {
  Region &thenRegion = kernel.getThenRegion();
  if (thenRegion.empty() || thenRegion.getBlocks().size() > 1)
    return false;
  Block &block = thenRegion.front();
  if (block.empty())
    return false;
  // We could relax this to be not the first op
  Operation &firstOp = block.front();
  auto ifOp = dyn_cast<scf::IfOp>(firstOp);
  if (!ifOp || !ifOp->hasAttr("db_prefetch"))
    return false;
  schedule.prefetch = ifOp;
  return true;
}

// Extract the store-back after kernel (ping or pong) compute.
bool extractStoreBack(scf::IfOp kernel, KernelSchedule &schedule) {
  // schedule.prefetch is already set by extractPreFetch
  if (!schedule.prefetch)
    return false;

  Operation *compute = schedule.prefetch->getNextNode();
  if (!compute)
    return false;

  if (!isa<scf::ForOp>(compute) && !isa<scf::ForallOp>(compute))
    return false;
  schedule.compute = compute;

  // After compute, collect memref::CopyOp till the yield
  Operation *currentOp = compute->getNextNode();
  while (currentOp) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(currentOp))
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
    auto attr = forOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
    if (attr && id == attr.getInt()) {
      foundPrologue = true;
      schedule.prologue = ifOp;
      break;
    }
  }
  if (!foundPrologue)
    return false;
  DBG("Prologue = " << schedule.prologue);

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
    registry.insert<scf::SCFDialect>();
    registry.insert<mlir::hexagonmem::HexagonMemDialect>();
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
