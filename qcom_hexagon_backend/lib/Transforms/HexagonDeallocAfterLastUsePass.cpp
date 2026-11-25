//===- HexagonDeallocAfterLastUsePass.cpp - dealloc after last use --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// HexagonDeallocAfterLastUsePass: standalone deallocation inserter for large
// memref.alloc buffers whose lifetime crosses outlined function calls.
//
// The stock buffer-deallocation pipeline cannot handle this case because its
// BufferViewFlowAnalysis is intra-function only: at a call it conservatively
// assumes every operand aliases every result and never inspects the callee to
// learn that e.g. `return %arg2` passes an argument straight through to a
// result. This pass builds that interprocedural pass-through fact explicitly.
//
// For each eligible memref.alloc, hexagonmem.alloc, or owned call result:
//   1. Forward-trace every alias of the buffer through view ops
//      (subview / expand_shape / collapse_shape / cast).
//   2. When an alias is passed to a call whose callee returns that exact
//      argument (return %argN), track the corresponding call RESULT as a
//      further alias — the buffer flows back out through the return.
//   3. Keep DMA source/destination buffers live through their matching
//      memref.dma_wait.
//   4. Find the last program point that touches any alias.
//   5. Insert or move the matching dealloc immediately after it.
//
// Aggressive: when the buffer's fate past a call cannot be proven a
// pass-through, it is assumed dead after that call. The only hard exclusion is
// a buffer that escapes via the entry function's own return.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

using namespace mlir;

#define GEN_PASS_DEF_HEXAGONDEALLOCAFTERLASTUSE
namespace mlir {
namespace hexagon {
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

// Byte size of a static int/float memref, or nullopt if not computable.
static std::optional<int64_t> staticByteSize(Type type) {
  auto mt = dyn_cast<MemRefType>(type);
  if (!mt || !mt.hasStaticShape())
    return std::nullopt;
  auto elemType = mt.getElementType();
  if (!elemType.isIntOrFloat())
    return std::nullopt;
  unsigned bw = elemType.getIntOrFloatBitWidth();
  if (bw == 0)
    return std::nullopt;
  return mt.getNumElements() * ((bw + 7) / 8);
}

static bool isVTCMMemref(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getMemorySpaceAsInt() == 1;
}

// The `all_parallel` marker is used on structured scf.for nests that are
// lowered as parallel tiled work. Treat those regions like scf.forall and
// scf.parallel: lexical last use is not sufficient to prove that no worker
// still accesses the buffer. The existing dealloc placement is retained in
// that case.
static bool isInsideConcurrentRegion(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp()) {
    if (isa<scf::ForallOp, scf::ParallelOp>(parent))
      return true;
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      if (forOp->hasAttr("all_parallel"))
        return true;
  }
  return false;
}

// A callee result index that is a straight pass-through of a callee argument.
// Key = (callee funcOp, result index) -> argument index it forwards.
using PassThroughMap =
    llvm::DenseMap<std::pair<Operation *, unsigned>, unsigned>;

// Set of (callee funcOp, result index) whose returned value is an OWNED buffer:
// it traces back (through view ops) to a memref.alloc or hexagonmem.alloc in
// the callee, or
// transitively to another owned-alloc call result. Such a result transfers
// ownership of a fresh buffer to the caller, which becomes responsible for
// deallocating it.
using OwnedReturnSet = llvm::DenseSet<std::pair<Operation *, unsigned>>;

// Trace a memref value backward through the view ops the pass understands
// (cast / subview / collapse_shape / expand_shape) to the value the view chain
// was built from. Stops at an alloc, a call result, a block argument, or any
// other defining op.
static Value traceBackThroughViews(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<memref::CastOp, memref::ViewOp, memref::SubViewOp,
            memref::CollapseShapeOp, memref::ExpandShapeOp,
            memref::ReinterpretCastOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

} // namespace

struct HexagonDeallocAfterLastUsePass
    : public mlir::hexagon::impl::HexagonDeallocAfterLastUseBase<
          HexagonDeallocAfterLastUsePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    hexagonmem::HexagonMemDialect>();
  }

  // Build the interprocedural pass-through facts for every func in the module:
  // a callee result r that returns entry-block argument a becomes
  // passthrough[(callee, r)] = a. Only single-block, single-return functions
  // whose return operand is a plain entry BlockArgument qualify.
  static PassThroughMap buildPassThroughMap(ModuleOp module) {
    PassThroughMap map;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() || !func.getBody().hasOneBlock())
        continue;
      Block &entry = func.getBody().front();
      auto ret = dyn_cast<func::ReturnOp>(entry.getTerminator());
      if (!ret)
        continue;
      for (auto [r, operand] : llvm::enumerate(ret.getOperands())) {
        auto ba = dyn_cast<BlockArgument>(operand);
        if (!ba || ba.getOwner() != &entry)
          continue;
        map[{func.getOperation(), (unsigned)r}] = ba.getArgNumber();
      }
    }
    return map;
  }

  // Classify each (callee, result index) whose returned value is an owned
  // buffer: it traces back through view ops to a memref.alloc in the callee, or
  // (transitively) to another call result that is itself an owned return.
  // Iterated to a fixpoint so ownership propagates up through chains of
  // returning callees.
  static OwnedReturnSet buildOwnedReturnSet(ModuleOp module,
                                            const PassThroughMap &passthrough) {
    OwnedReturnSet owned;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto func : module.getOps<func::FuncOp>()) {
        if (func.isExternal() || !func.getBody().hasOneBlock())
          continue;
        Block &entry = func.getBody().front();
        auto ret = dyn_cast<func::ReturnOp>(entry.getTerminator());
        if (!ret)
          continue;
        for (auto [r, operand] : llvm::enumerate(ret.getOperands())) {
          std::pair<Operation *, unsigned> key{func.getOperation(),
                                               (unsigned)r};
          if (owned.contains(key))
            continue;
          // A pure argument pass-through is NOT an owned alloc (handled
          // separately by PassThroughMap — ownership stays with the caller's
          // original buffer).
          Value root = traceBackThroughViews(operand);
          bool isOwned = false;
          if (root.getDefiningOp<memref::AllocOp>() ||
              root.getDefiningOp<hexagonmem::AllocOp>()) {
            isOwned = true;
          } else if (auto call = root.getDefiningOp<func::CallOp>()) {
            auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
            if (callee) {
              unsigned ri = cast<OpResult>(root).getResultNumber();
              if (owned.contains({callee.getOperation(), ri}))
                isOwned = true;
            }
          }
          if (isOwned) {
            owned.insert(key);
            changed = true;
          }
        }
      }
    }
    return owned;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassThroughMap passthrough = buildPassThroughMap(module);
    OwnedReturnSet ownedReturns = buildOwnedReturnSet(module, passthrough);

    // Process every function body: the public entry AND private outlined
    // callees (e.g. @__outlined_generic_*), each of which may hold its own
    // local memref.alloc buffers. dealloc_helper functions are skipped so we
    // never inject deallocs into the deallocation machinery itself.
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() || func.getName().contains("dealloc_helper"))
        continue;
      processFunction(func, passthrough, ownedReturns);
    }
  }

  // Insert or move the matching dealloc after the last use of each owned
  // buffer in `func`. Owned buffers are local memref/hexagonmem allocations
  // and results of calls whose callee allocates-and-returns a buffer
  // (ownership transferred to this frame). Escape is judged against `func`'s
  // own return; a buffer returned by `func` is never deallocated here.
  void processFunction(func::FuncOp func, const PassThroughMap &passthrough,
                       const OwnedReturnSet &ownedReturns) {
    ModuleOp module = func->getParentOfType<ModuleOp>();
    Block &entry = func.getBody().front();

    // Number every op in this function in pre-order (parent before child,
    // siblings in program order) — same idiom as VTCMAllocUtils.
    llvm::DenseMap<Operation *, int> opToId;
    {
      int cursor = 0;
      func.walk<WalkOrder::PreOrder>(
          [&](Operation *op) { opToId[op] = cursor++; });
    }
    auto ret = dyn_cast<func::ReturnOp>(entry.getTerminator());

    // Collect owned-buffer roots in program order:
    //   - every static memref.alloc above the configured threshold, or in
    //     VTCM;
    //   - every static hexagonmem.alloc; and
    //   - every owned call result above the same threshold, or in VTCM.
    // The Value is the buffer we forward-trace and ultimately dealloc.
    SmallVector<Value> roots;
    func.walk([&](Operation *op) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        auto bytes = staticByteSize(alloc.getResult().getType());
        if (bytes &&
            (isVTCMMemref(alloc.getResult().getType()) || *bytes >= chunkBytes))
          roots.push_back(alloc.getResult());
        return;
      }
      if (auto alloc = dyn_cast<hexagonmem::AllocOp>(op)) {
        if (staticByteSize(alloc.getResult().getType()))
          roots.push_back(alloc.getResult());
        return;
      }
      if (auto call = dyn_cast<func::CallOp>(op)) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee)
          return;
        for (auto [r, result] : llvm::enumerate(call.getResults())) {
          if (!ownedReturns.contains({callee.getOperation(), (unsigned)r}))
            continue;
          auto bytes = staticByteSize(result.getType());
          if (bytes && (isVTCMMemref(result.getType()) || *bytes >= chunkBytes))
            roots.push_back(result);
        }
        return;
      }
      // scf.if results that own an allocation: when an alloc is created
      // inside a branch and yielded out, the allocation's lifetime is
      // controlled by the scf.if result in the outer scope. Add that result
      // as a root so a dealloc is placed after its last use outside the if.
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto isOwnedYield = [&](Region &region, unsigned r) {
          if (!region.hasOneBlock())
            return false;
          auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
          if (!yield || r >= yield.getNumOperands())
            return false;
          Value base = traceBackThroughViews(yield.getOperand(r));
          return base.getDefiningOp<memref::AllocOp>() != nullptr ||
                 base.getDefiningOp<hexagonmem::AllocOp>() != nullptr;
        };
        for (auto result : ifOp.getResults()) {
          if (!isa<MemRefType>(result.getType()))
            continue;
          unsigned r = result.getResultNumber();
          if (!isOwnedYield(ifOp.getThenRegion(), r) &&
              !isOwnedYield(ifOp.getElseRegion(), r))
            continue;
          // Use the base alloc's size for the threshold check.
          auto getBase = [&](Region &region, unsigned idx) -> Value {
            if (!region.hasOneBlock())
              return {};
            auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
            if (!yield || idx >= yield.getNumOperands())
              return {};
            return traceBackThroughViews(yield.getOperand(idx));
          };
          Value base = getBase(ifOp.getThenRegion(), r);
          if (!base)
            base = getBase(ifOp.getElseRegion(), r);
          if (!base)
            continue;
          auto bytes = staticByteSize(base.getType());
          if (bytes && (isVTCMMemref(base.getType()) || *bytes >= chunkBytes))
            roots.push_back(result);
        }
      }
    });

    OpBuilder b(module.getContext());
    BufferViewFlowAnalysis aliases(func);

    auto isDealloc = [](Operation *op) {
      return isa<memref::DeallocOp, hexagonmem::DeallocOp>(op);
    };

    auto sameAlias = [&](Value lhs, Value rhs) {
      if (lhs == rhs)
        return true;
      for (Value alias : aliases.resolve(lhs))
        if (alias == rhs)
          return true;
      for (Value alias : aliases.resolve(rhs))
        if (alias == lhs)
          return true;
      return false;
    };

    for (Value root : roots) {
      Operation *rootOp = root.getDefiningOp();
      if (!rootOp)
        continue;

      SmallVector<Value> aliasValues;
      for (Value alias : aliases.resolve(root))
        if (!llvm::is_contained(aliasValues, alias))
          aliasValues.push_back(alias);
      if (!llvm::is_contained(aliasValues, root))
        aliasValues.push_back(root);

      SmallVector<Operation *, 2> existingDeallocs;
      for (Value alias : aliasValues) {
        for (Operation *user : alias.getUsers()) {
          if (isDealloc(user) && !llvm::is_contained(existingDeallocs, user))
            existingDeallocs.push_back(user);
        }
      }

      // Forward alias walk. Track the last-use op (max pre-order id) across
      // all aliases; detect escape via this function's return. DMA sources and
      // destinations remain live through the first matching wait on the same
      // tag.
      llvm::DenseSet<Value> visited;
      SmallVector<Value> worklist{root};
      Operation *lastUseOp = rootOp;
      int lastUseId = opToId.lookup(rootOp);
      bool escapes = false;
      bool unknownAsyncUse = false;
      // An allocation created inside a concurrent region cannot be moved out
      // of that region: each worker may own a distinct dynamic instance.
      // An allocation created outside the region can instead be freed after
      // the enclosing operation completes, which is its join point.
      bool allocationInsideConcurrentRegion = isInsideConcurrentRegion(rootOp);
      bool hasSynchronousUse = false;
      SmallVector<memref::DmaStartOp> loopCarriedPrefetches;

      auto consider = [&](Operation *user) {
        auto it = opToId.find(user);
        if (it != opToId.end() && it->second > lastUseId) {
          lastUseId = it->second;
          lastUseOp = user;
        }
      };

      auto findMatchingDmaWait = [&](memref::DmaStartOp start) {
        Operation *matchingWait = nullptr;
        int matchingWaitId = std::numeric_limits<int>::max();
        int startId = opToId.lookup(start.getOperation());
        Value startTag = traceBackThroughViews(start.getTagMemRef());
        func.walk([&](memref::DmaWaitOp wait) {
          int waitId = opToId.lookup(wait.getOperation());
          if (waitId <= startId)
            return;
          Value waitTag = traceBackThroughViews(wait.getTagMemRef());
          if (startTag != waitTag &&
              !sameAlias(start.getTagMemRef(), wait.getTagMemRef()))
            return;
          if (waitId < matchingWaitId) {
            matchingWait = wait.getOperation();
            matchingWaitId = waitId;
          }
        });
        return matchingWait;
      };

      // Double-buffer insertion deliberately places a prefetch in one loop
      // iteration and waits for that tag in the next iteration. The final
      // prefetch is guarded by the "next tile exists" condition, so it cannot
      // execute on the last iteration. Recognize only this explicit protocol;
      // a generic wait-before-start loop remains unknown.
      auto hasLoopCarriedPrefetchWait = [&](memref::DmaStartOp start) {
        bool isPrefetch = false;
        for (Operation *parent = start.getOperation()->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (auto ifOp = dyn_cast<scf::IfOp>(parent))
            isPrefetch |= ifOp->hasAttr("db_prefetch");
          if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            if (!isPrefetch || !loop->hasAttr("db_generic"))
              return false;

            Value startTag = traceBackThroughViews(start.getTagMemRef());
            bool foundWait = false;
            loop.walk([&](memref::DmaWaitOp wait) {
              Value waitTag = traceBackThroughViews(wait.getTagMemRef());
              if (startTag == waitTag ||
                  sameAlias(start.getTagMemRef(), wait.getTagMemRef()))
                foundWait = true;
            });
            return foundWait;
          }
        }
        return false;
      };

      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (!visited.insert(v).second)
          continue;

        for (OpOperand &use : v.getUses()) {
          Operation *user = use.getOwner();

          // Escape: buffer reaches this function's return.
          if (ret && user == ret.getOperation()) {
            escapes = true;
            continue;
          }

          // Escape via scf.yield: the buffer is yielded out of a region
          // (e.g. an scf.if branch). The scf.if result is handled as its
          // own root (see roots collection above), so we must not dealloc
          // the inner alloc inside the branch.
          if (isa<scf::YieldOp>(user)) {
            escapes = true;
            continue;
          }

          if (isDealloc(user))
            continue;

          consider(user);

          if (auto dmaStart = dyn_cast<memref::DmaStartOp>(user)) {
            if (Operation *wait = findMatchingDmaWait(dmaStart))
              consider(wait);
            else if (hasLoopCarriedPrefetchWait(dmaStart))
              loopCarriedPrefetches.push_back(dmaStart);
            else
              unknownAsyncUse = true;
          } else if (!isa<memref::CastOp, memref::ViewOp, memref::SubViewOp,
                          memref::CollapseShapeOp, memref::ExpandShapeOp,
                          memref::ReinterpretCastOp, memref::DmaWaitOp>(user)) {
            // A synchronous access is evidence that this buffer is consumed
            // by the double-buffer protocol after its prefetch completes.
            hasSynchronousUse = true;
          }

          // When the buffer's raw pointer is extracted via
          // extract_aligned_pointer_as_index, the resulting index value (and
          // any further integer/pointer casts derived from it) are raw aliases
          // of the underlying memory. Trace through the cast chain so that the
          // last actual memory-accessing op (e.g. an llvm.call that receives
          // the converted pointer) is correctly identified as the last use.
          if (isa<memref::ExtractAlignedPointerAsIndexOp>(user)) {
            SmallVector<Value> ptrWorklist(user->result_begin(),
                                           user->result_end());
            DenseSet<Value> ptrVisited;
            while (!ptrWorklist.empty()) {
              Value pv = ptrWorklist.pop_back_val();
              if (!ptrVisited.insert(pv).second)
                continue;
              for (OpOperand &pu : pv.getUses()) {
                Operation *puser = pu.getOwner();
                consider(puser);
                if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
                        LLVM::IntToPtrOp, LLVM::PtrToIntOp>(puser))
                  for (Value res : puser->getResults())
                    ptrWorklist.push_back(res);
              }
            }
          }

          // View ops produce a new alias of the same buffer.
          if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
            worklist.push_back(sv.getResult());
            continue;
          }
          if (auto view = dyn_cast<memref::ViewOp>(user)) {
            worklist.push_back(view.getResult());
            continue;
          }
          if (auto ex = dyn_cast<memref::ExpandShapeOp>(user)) {
            worklist.push_back(ex.getResult());
            continue;
          }
          if (auto co = dyn_cast<memref::CollapseShapeOp>(user)) {
            worklist.push_back(co.getResult());
            continue;
          }
          if (auto ca = dyn_cast<memref::CastOp>(user)) {
            worklist.push_back(ca.getResult());
            continue;
          }
          if (auto rc = dyn_cast<memref::ReinterpretCastOp>(user)) {
            worklist.push_back(rc.getResult());
            continue;
          }

          // Interprocedural pass-through: if the callee returns this exact
          // argument, the buffer flows out through the call result. (Owned
          // returns are handled as their own roots, not traced from here.)
          if (auto call = dyn_cast<func::CallOp>(user)) {
            auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
            if (!callee)
              continue;
            unsigned p = use.getOperandNumber();
            for (auto [r, result] : llvm::enumerate(call.getResults())) {
              auto ptIt =
                  passthrough.find({callee.getOperation(), (unsigned)r});
              if (ptIt != passthrough.end() && ptIt->second == p)
                worklist.push_back(result);
            }
          }
        }
      }

      // A loop-carried prefetch is only safe to discharge here when the
      // buffer is also consumed synchronously. Otherwise a buffer used solely
      // by an unobserved asynchronous transfer would still be unknown.
      if (!loopCarriedPrefetches.empty() && !hasSynchronousUse)
        unknownAsyncUse = true;

      // Escaping buffers are never deallocated.
      // If an async transfer has no provable wait, retain any existing
      // dealloc rather than introducing or moving one before completion.
      if (escapes || unknownAsyncUse || allocationInsideConcurrentRegion)
        continue;

      // Place the dealloc in the allocation's block. For an allocation in a
      // loop body this keeps the lifetime loop-local; for an entry-block
      // allocation this naturally produces a function-scope dealloc.
      Operation *anchor = lastUseOp;
      Block *allocBlock = rootOp->getBlock();
      while (anchor && anchor->getBlock() != allocBlock) {
        Operation *parent = anchor->getParentOp();
        if (!parent)
          break;
        anchor = parent;
      }
      if (!anchor || anchor->getBlock() != allocBlock)
        continue;

      bool useHexagonMemDealloc =
          isa<hexagonmem::AllocOp>(rootOp) || isVTCMMemref(root.getType()) ||
          llvm::any_of(existingDeallocs, [](Operation *op) {
            return isa<hexagonmem::DeallocOp>(op);
          });

      // A single existing dealloc is moved by recreating it at the new
      // insertion point. Multiple control-flow-specific deallocs are
      // intentionally left untouched because moving one would not preserve
      // their path semantics.
      if (existingDeallocs.size() > 1)
        continue;
      for (Operation *dealloc : existingDeallocs)
        dealloc->erase();

      b.setInsertionPointAfter(anchor);
      if (useHexagonMemDealloc)
        hexagonmem::DeallocOp::create(b, root.getLoc(), root);
      else
        memref::DeallocOp::create(b, root.getLoc(), root);
    }
  }
};

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass>
createHexagonDeallocAfterLastUsePass(int64_t chunkBytesVal) {
  HexagonDeallocAfterLastUseOptions opts;
  opts.chunkBytes = chunkBytesVal;
  return std::make_unique<HexagonDeallocAfterLastUsePass>(opts);
}

} // namespace hexagon
} // namespace mlir
